from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int, encoding_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim * 8),
            nn.ReLU(),
            nn.Linear(encoding_dim * 8, encoding_dim * 4),
            nn.ReLU(),
            nn.Linear(encoding_dim * 4, encoding_dim * 2),
            nn.ReLU(),
            nn.Linear(encoding_dim * 2, encoding_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, input_dim * 4),
            nn.ReLU(),
            nn.Linear(input_dim * 4, input_dim * 8),
            nn.ReLU(),
            nn.Linear(input_dim * 8, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, output_dim),
        )

    def forward(self, x):
        return self.layers(x)


class CleanModel:
    def __init__(self, input_dim: int, output_dim: int, device: torch.device):
        self.model1 = MLP(input_dim, output_dim).to(device)
        self.model2 = MLP(input_dim, output_dim).to(device)
        self.optimizer1 = torch.optim.AdamW(self.model1.parameters())
        self.optimizer2 = torch.optim.AdamW(self.model2.parameters())
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.device = device

    @staticmethod
    def _schedule_forget_rate(current_epoch: int, num_epochs: int, max_forget_rate: float) -> float:
        if current_epoch <= 3:
            return 0
        return min(max_forget_rate * current_epoch / num_epochs + 0.03, max_forget_rate)

    def train(
        self,
        x_train,
        y_train,
        x_val=None,
        y_val=None,
        max_forget_rate: float = 0.3,
        num_epochs: int = 100,
        batch_size: int = 128,
        patience: int = 10,
    ):
        dataset = torch.utils.data.TensorDataset(x_train, y_train)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        best_val_loss = float("inf")
        early_stop_counter = 0

        for epoch in range(num_epochs):
            self.model1.train()
            self.model2.train()

            forget_rate = self._schedule_forget_rate(epoch, num_epochs, max_forget_rate)
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)

                logits1 = self.model1(x)
                logits2 = self.model2(x)
                loss1 = self.criterion(logits1, y)
                loss2 = self.criterion(logits2, y)

                idx1 = torch.argsort(loss1)[: int((1 - forget_rate) * len(loss1))]
                idx2 = torch.argsort(loss2)[: int((1 - forget_rate) * len(loss2))]

                loss_update1 = torch.mean(loss1[idx2])
                loss_update2 = torch.mean(loss2[idx1])

                self.optimizer1.zero_grad()
                loss_update1.backward()
                self.optimizer1.step()

                self.optimizer2.zero_grad()
                loss_update2.backward()
                self.optimizer2.step()

            if x_val is not None and y_val is not None:
                self.model1.eval()
                self.model2.eval()
                with torch.no_grad():
                    val_logits1 = self.model1(x_val.to(self.device))
                    val_logits2 = self.model2(x_val.to(self.device))
                    val_loss1 = nn.CrossEntropyLoss()(val_logits1, y_val.to(self.device)).item()
                    val_loss2 = nn.CrossEntropyLoss()(val_logits2, y_val.to(self.device)).item()
                    val_loss_avg = (val_loss1 + val_loss2) / 2

                if val_loss_avg < best_val_loss:
                    best_val_loss = val_loss_avg
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1

                if epoch % 10 == 0 or epoch == num_epochs - 1:
                    print(
                        f"[Epoch {epoch + 1}] Forget rate: {forget_rate:.3f}, "
                        f"Val Loss1: {val_loss1:.4f}, Val Loss2: {val_loss2:.4f}"
                    )
                elif early_stop_counter >= patience:
                    print(
                        f"[Epoch {epoch + 1}] Forget rate: {forget_rate:.3f}, "
                        f"Val Loss1: {val_loss1:.4f}, Val Loss2: {val_loss2:.4f}"
                    )
                    print("Early stopping triggered.")
                    break

    def predict(self, x):
        tau = 1.5
        self.model1.eval()
        self.model2.eval()
        with torch.no_grad():
            logits1 = self.model1(x.to(self.device))
            logits2 = self.model2(x.to(self.device))
            probs1 = torch.softmax(logits1 / tau, dim=1)
            probs2 = torch.softmax(logits2 / tau, dim=1)
            avg_probs = (probs1 + probs2) / 2
            confs, preds = torch.max(avg_probs, dim=1)
        return preds.cpu(), confs.cpu()


class NoisyModel(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.transition_matrix_layer = nn.Linear(256, num_classes * num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        flat_transition_matrix = self.transition_matrix_layer(features)
        transition_matrix = flat_transition_matrix.view(-1, self.num_classes, self.num_classes)
        return F.softmax(transition_matrix, dim=2)
