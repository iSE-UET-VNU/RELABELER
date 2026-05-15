from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset, random_split

from core.utils import ensure_dir
from correction.bayesian_inference import infer_clean_label
from models import MLP, NoisyModel


class LabelCorrector:
    def __init__(self, num_classes: int, device: torch.device, model_dir: str | Path = "models"):
        self.num_classes = num_classes
        self.device = device
        self.model_dir = ensure_dir(model_dir)
        self.noisy_model = None
        self.simulator_model = None

    def _train_simulator_model(self, x_corrupted, y_corrupted, epochs: int = 100, batch_size: int = 128):
        print("\n--- Training simulator_model on Dcorrupted ---")
        input_dim = x_corrupted.shape[1]

        self.simulator_model = MLP(input_dim, self.num_classes).to(self.device)
        optimizer = torch.optim.AdamW(self.simulator_model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        dataset = TensorDataset(
            torch.tensor(x_corrupted, dtype=torch.float32),
            torch.tensor(y_corrupted, dtype=torch.long),
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.simulator_model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = self.simulator_model(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if (epoch + 1) % 20 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(loader):.4f}")
        print("Finished training simulator_model.")

    def _train_noisy_model(self, x_trans, y_clean_trans, y_noise_trans, epochs: int = 100, batch_size: int = 128):
        print("\n--- Training noisy_model on Dtrans ---")
        input_dim = x_trans.shape[1]

        self.noisy_model = NoisyModel(input_dim, self.num_classes).to(self.device)
        optimizer = torch.optim.AdamW(self.noisy_model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)

        dataset = TensorDataset(
            torch.tensor(x_trans, dtype=torch.float32),
            torch.tensor(y_clean_trans, dtype=torch.long),
            torch.tensor(y_noise_trans, dtype=torch.long),
        )

        val_size = max(1, int(0.15 * len(dataset))) if len(dataset) > 1 else 0
        train_size = len(dataset) - val_size
        if val_size > 0:
            train_subset, val_subset = random_split(dataset, [train_size, val_size])
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        else:
            train_subset = dataset
            val_loader = None
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

        print(
            f"Training on {len(train_subset)} samples, "
            f"validating on {len(val_loader.dataset) if val_loader else 0} samples."
        )

        best_val_loss = float("inf")
        epochs_no_improve = 0
        patience = 10
        best_model_path = self.model_dir / "best_noisy_model.pth"

        for epoch in range(epochs):
            self.noisy_model.train()
            for x_batch, y_clean_batch, y_noise_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_clean_batch = y_clean_batch.to(self.device)
                y_noise_batch = y_noise_batch.to(self.device)
                optimizer.zero_grad()

                transition_matrix = self.noisy_model(x_batch)
                predicted_noise_dist = transition_matrix[torch.arange(x_batch.size(0)), y_clean_batch]

                loss = criterion(predicted_noise_dist, y_noise_batch)
                loss.backward()
                optimizer.step()

            if val_loader is None:
                continue

            self.noisy_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_batch, y_clean_batch, y_noise_batch in val_loader:
                    x_batch = x_batch.to(self.device)
                    y_clean_batch = y_clean_batch.to(self.device)
                    y_noise_batch = y_noise_batch.to(self.device)
                    transition_matrix = self.noisy_model(x_batch)
                    predicted_noise_dist = transition_matrix[torch.arange(x_batch.size(0)), y_clean_batch]
                    val_loss += criterion(predicted_noise_dist, y_noise_batch).item()

            avg_val_loss = val_loss / len(val_loader)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Val Loss: {avg_val_loss:.4f}")

            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                torch.save(self.noisy_model.state_dict(), best_model_path)
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

        if best_model_path.exists():
            print(f"Finished training noisy_model. Loading best model with validation loss: {best_val_loss:.4f}")
            self.noisy_model.load_state_dict(torch.load(best_model_path, map_location=self.device))

    def correct_labels(
        self,
        train_df: pd.DataFrame,
        clean_indices,
        corrupted_indices,
        y_pred_clean_soft: np.ndarray,
        confidence_threshold: float = 0.9,
    ) -> pd.DataFrame:
        print("\n--- Starting Label Correction Process ---")

        df_clean = train_df.loc[clean_indices]
        df_corrupted = train_df.loc[corrupted_indices]

        x_clean = df_clean.iloc[:, :-1].values
        y_clean = df_clean.iloc[:, -1].values
        x_corrupted = df_corrupted.iloc[:, :-1].values
        y_corrupted = df_corrupted.iloc[:, -1].values

        print("\nStep 2.1: Training simulator_model on Dcorrupted.")
        self._train_simulator_model(x_corrupted, y_corrupted)

        print("\nStep 2.2: Creating Dtrans by predicting on Dclean with simulator_model.")
        if self.simulator_model is None:
            raise RuntimeError("simulator_model was not trained.")

        self.simulator_model.eval()
        with torch.no_grad():
            outputs = self.simulator_model(torch.tensor(x_clean, dtype=torch.float32).to(self.device))
            _, y_noise_on_clean = torch.max(outputs, 1)
            y_noise_on_clean = y_noise_on_clean.cpu().numpy()

        x_trans, y_clean_trans, y_noise_trans = x_clean, y_clean, y_noise_on_clean
        print(f"Created Dtrans with {len(x_trans)} samples.")

        print("\nStep 3: Training noisy_model on Dtrans for correction scoring.")
        self._train_noisy_model(x_trans, y_clean_trans, y_noise_trans)

        print("\nStep 4: Correcting labels in Dcorrupted with the correction scoring rule.")
        if self.noisy_model is None:
            raise RuntimeError("noisy_model was not trained.")

        self.noisy_model.eval()
        corrected_labels_df = train_df.copy()

        with torch.no_grad():
            x_corrupted_tensor = torch.tensor(x_corrupted, dtype=torch.float32).to(self.device)
            transition_matrix = self.noisy_model(x_corrupted_tensor)
            p_y_clean_given_x = torch.tensor(y_pred_clean_soft, dtype=torch.float32).to(self.device)

            corrected_count = 0
            for i in range(len(df_corrupted)):
                y_c = y_corrupted[i]
                _, new_label, max_prob = infer_clean_label(
                    transition_matrix_for_sample=transition_matrix[i],
                    noisy_label=y_c,
                    clean_label_prior=p_y_clean_given_x[i],
                )

                if new_label != y_c and max_prob > confidence_threshold:
                    original_df_index = corrupted_indices[i]
                    corrected_labels_df.loc[original_df_index, "labels"] = new_label
                    corrected_count += 1

        print(f"\nFinished Correction. {corrected_count} labels were changed in the corrupted set.")
        return corrected_labels_df
