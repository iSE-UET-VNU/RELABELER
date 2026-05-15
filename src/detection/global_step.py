from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from detection.state import DetectionState
from evaluation import print_detection_results
from models import CleanModel


def global_step(
    state: DetectionState,
    n_iterations: int,
    confidence_threshold: float = 1,
    num_epochs: int = 200,
    seed: int = 2025,
) -> CleanModel | None:
    if state.detected_noisy_indices is None:
        raise ValueError("Run local_step first or provide state.detected_noisy_indices.")

    unlimited = n_iterations == -1
    if unlimited:
        n_iterations = 999
        previous_noisy_count = len(state.detected_noisy_indices)

    clean_model = None
    for iteration in range(n_iterations):
        print(f"\n--- Starting Global Detection Step Iteration {iteration + 1} ---")

        if state.detected_noisy_indices is None or len(state.detected_noisy_indices) == 0:
            print("No noisy indices detected or provided. Stopping.")
            break

        valid_indices_for_drop = state.train_df.index.intersection(state.detected_noisy_indices)
        df_clean = state.train_df.drop(valid_indices_for_drop)

        if len(df_clean) < 2 or len(df_clean["labels"].unique()) < 2:
            print("Not enough clean data or classes to train. Stopping.")
            break

        x = df_clean.iloc[:, :-1].values
        y = df_clean.iloc[:, -1].values

        if len(df_clean) <= 5:
            print(f"Skipping iteration {iteration + 1}: Not enough clean data for train/val split.")
            break

        try:
            x_train, x_val, y_train, y_val = train_test_split(
                x,
                y,
                test_size=0.2,
                stratify=y,
                random_state=seed,
            )
        except ValueError:
            print("Warning: Could not stratify train/val split. Proceeding without stratification.")
            x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=seed)

        current_noisy_indices = np.array(list(set(state.train_df.index) & set(state.detected_noisy_indices)))
        if len(current_noisy_indices) == 0:
            print(f"Iteration {iteration + 1}: No indices left to test. Stopping.")
            break

        df_test = state.train_df.loc[current_noisy_indices]
        x_test = df_test.iloc[:, :-1].values
        y_test = df_test.iloc[:, -1].values

        x_train = torch.tensor(x_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        x_val = torch.tensor(x_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.long)
        x_test = torch.tensor(x_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.long)

        clean_model = CleanModel(x_train.shape[1], state.num_labels, state.device)
        clean_model.train(
            x_train,
            y_train,
            x_val,
            y_val,
            max_forget_rate=state.test_forget_rate,
            num_epochs=num_epochs,
        )

        indices_to_keep_as_noisy = []
        high_confidence_noisy = 0
        pred_labels, pred_confidences = clean_model.predict(x_test)

        for i, original_df_index in enumerate(current_noisy_indices):
            current_label = y_test[i].item()
            predicted_label = pred_labels[i].item()
            confidence = pred_confidences[i].item()

            if predicted_label != current_label:
                indices_to_keep_as_noisy.append(original_df_index)
                if confidence >= confidence_threshold:
                    high_confidence_noisy += 1

        print(
            f"Iteration {iteration + 1}: Kept {len(indices_to_keep_as_noisy)} suspected noisy labels. "
            f"{high_confidence_noisy} have confidence >= {confidence_threshold}."
        )

        state.detected_noisy_indices = np.array(indices_to_keep_as_noisy)
        state.test_forget_rate = print_detection_results(
            train_df=state.train_df,
            clean_labels=state.clean_labels,
            corrupted_indices=state.corrupted_indices,
            detected_noisy_indices=state.detected_noisy_indices,
            phase="global",
            iteration=iteration,
        )

        if unlimited:
            current_noisy_count = len(state.detected_noisy_indices)
            if current_noisy_count >= previous_noisy_count - 1:
                print(
                    "Convergence detected: No significant reduction in noisy indices "
                    f"({previous_noisy_count} -> {current_noisy_count}). Stopping."
                )
                break
            previous_noisy_count = current_noisy_count
        elif len(state.detected_noisy_indices) == 0:
            print("No more noisy indices detected. Stopping.")
            break

    return clean_model
