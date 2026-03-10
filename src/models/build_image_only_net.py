import torch
import torch.nn as nn


class ImageOnlyNet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

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

        # ---------------- Classifier ----------------
        self.classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, img):
        f2d = self.fc_2d(self.branch_2d(img))
        return self.classifier(f2d)
