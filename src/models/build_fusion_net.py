import torch
import torch.nn as nn
import torch.nn.functional as F


class MultimodalFusionNet(nn.Module):
    """
    Multi-task fusion network.

    Inputs : image (B, 1, H, W)  +  signal (B, 2, 2000)
    Outputs (dict):
        class_logits  (B, 6)    — transfer function class
        dt_logit      (B,)      — has dead time  (raw logit for BCEWithLogitsLoss)
        dt_value      (B,)      — dead time magnitude  (normalised, regression)
        reg           (B, 4)    — [settling_time, rise_time, overshoot_pct, steady_state_gain]
                                   all normalised to ~[0, 1]
    """

    def __init__(self, num_classes=6, num_reg_targets=4):
        super().__init__()

        # ── Signal branch (1D CNN) ──────────────────────────────
        self.branch_1d = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, padding='same', bias=False),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding='same', bias=False),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm1d(256), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.fc_1d = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4),
        )

        # ── Image branch (2D CNN) ───────────────────────────────
        self.branch_2d = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding='same', bias=False),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding='same', bias=False),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding='same', bias=False),
            nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.fc_2d = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.4),
        )

        # ── Shared fusion layer (128 + 128 = 256 → 128) ────────
        self.fusion = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
        )

        # ── Output heads ────────────────────────────────────────
        self.head_class  = nn.Linear(128, num_classes)  # 6-way classification
        self.head_dt_cls = nn.Linear(128, 1)            # has dead time (binary)
        self.head_dt_val = nn.Linear(128, 1)            # dead time value
        self.head_reg    = nn.Sequential(               # settling, rise, overshoot, gain
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, num_reg_targets),
        )

    def forward(self, img, sig):
        f1d = F.normalize(self.fc_1d(self.branch_1d(sig)), dim=1)
        f2d = F.normalize(self.fc_2d(self.branch_2d(img)), dim=1)
        feat = self.fusion(torch.cat([f1d, f2d], dim=1))

        return {
            'class_logits': self.head_class(feat),              # (B, 6)
            'dt_logit':     self.head_dt_cls(feat).squeeze(1),  # (B,)
            'dt_value':     self.head_dt_val(feat).squeeze(1),  # (B,)
            'reg':          self.head_reg(feat),                 # (B, 4)
        }
