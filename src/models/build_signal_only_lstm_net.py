import torch
import torch.nn as nn

class SignalOnlyNetLSTM(nn.Module):
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
            nn.AdaptiveAvgPool1d(50)  # 500 -> 50 timesteps (manageable for LSTM)
        )

        # ---------------- LSTM Layer ----------------
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
                # Set forget gate bias to 1.0 (hidden_size indices 64:128)
                param.data[64:128] = 1.0

        self.fc_1d = nn.Sequential(
            nn.Linear(128, 64),  # 64*2 (bidirectional)
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
        x = self.branch_1d(sig)
        
        # Permute from (Batch, Channels, SeqLen) to (Batch, SeqLen, Channels)
        x = x.permute(0, 2, 1)
        
        # LSTM output extraction
        lstm_out, _ = self.lstm(x)
        
        # Isolate final timestep hidden state
        x = lstm_out[:, -1, :]
        
        x = self.fc_1d(x)
        return self.classifier(x)