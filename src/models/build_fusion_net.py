import torch
import torch.nn as nn
import torch.nn.functional as F

class MultimodalFusionNet(nn.Module):
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

        # ---------------- 2D Branch (Image) ----------------
        self.branch_2d = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding='same', bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding='same', bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding='same', bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc_2d = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        # ---------------- Fusion Classifier ----------------
        # 64 (1D) + 64 (2D) = 128
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, img, sig):
        f1d = self.fc_1d(self.branch_1d(sig))
        f2d = self.fc_2d(self.branch_2d(img))

        # Normalize to prevent modality dominance
        f1d = F.normalize(f1d, dim=1)
        f2d = F.normalize(f2d, dim=1)

        combined = torch.cat((f1d, f2d), dim=1)
        return self.classifier(combined)