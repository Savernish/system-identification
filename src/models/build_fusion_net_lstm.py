import torch
import torch.nn as nn
import torch.nn.functional as F


class MultimodalFusionNetLSTM(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        # ---------------- 1D Branch (Signal + LSTM) ----------------
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
            nn.AdaptiveAvgPool1d(50)  # 500 -> 50 timesteps for LSTM
        )

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # Orthogonal init for LSTM weights + forget gate bias = 1.0
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name or 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                param.data[64:128] = 1.0

        self.fc_1d = nn.Sequential(
            nn.Linear(128, 64),  # 64*2 (bidirectional)
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
        # 64 (1D-LSTM) + 64 (2D) = 128
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, img, sig):
        x1d = self.branch_1d(sig)
        x1d = x1d.permute(0, 2, 1)       # (B, Channels, SeqLen) -> (B, SeqLen, Channels)
        lstm_out, _ = self.lstm(x1d)
        f1d = self.fc_1d(lstm_out[:, -1, :])  # last timestep

        f2d = self.fc_2d(self.branch_2d(img))

        # Normalize to prevent modality dominance
        f1d = F.normalize(f1d, dim=1)
        f2d = F.normalize(f2d, dim=1)

        combined = torch.cat((f1d, f2d), dim=1)
        return self.classifier(combined)