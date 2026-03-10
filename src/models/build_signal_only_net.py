import torch
import torch.nn as nn


class SignalOnlyNet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        # ---------------- 1D Branch (Signal) ----------------
        self.branch_1d = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, padding='same', bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding='same', bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.fc_1d = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        # ---------------- Classifier ----------------
        self.classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, sig):
        f1d = self.fc_1d(self.branch_1d(sig))
        return self.classifier(f1d)
