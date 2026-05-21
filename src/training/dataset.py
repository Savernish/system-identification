import os
import csv
import glob
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Normalisation constants — bring all regression targets into roughly [0, 1]
REG_SCALES = {
    'settling_time':     100.0,
    'rise_time':         100.0,
    'overshoot_pct':     100.0,
    'steady_state_gain':   5.0,
}
DT_VALUE_SCALE = 15.0   # max dead time (seconds)


class SystemIDDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.img_dir = os.path.join(root_dir, 'images', split)
        self.sig_dir = os.path.join(root_dir, 'signals', split)

        self.classes = sorted(
            [f.name for f in os.scandir(self.img_dir) if f.is_dir()]
        ) if os.path.exists(self.img_dir) else []
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # Build metadata lookup: (class, filename) -> row dict
        meta_path = os.path.join(root_dir, f'metadata_{split}.csv')
        metadata = {}
        if os.path.exists(meta_path):
            with open(meta_path, newline='') as f:
                for row in csv.DictReader(f):
                    metadata[(row['class'], row['filename'])] = row

        self.transform = transforms.Compose([
            transforms.Grayscale(1),
            transforms.ToTensor(),
        ])

        images, signals, labels = [], [], []
        has_dt_list, dt_val_list, reg_list = [], [], []

        for cls in self.classes:
            img_paths = sorted(glob.glob(os.path.join(self.img_dir, cls, '*.png')))
            for img_path in img_paths:
                fname    = os.path.splitext(os.path.basename(img_path))[0]
                sig_path = os.path.join(self.sig_dir, cls, fname + '.npy')
                if not os.path.exists(sig_path):
                    continue

                with Image.open(img_path) as im:
                    images.append(self.transform(im))
                signals.append(torch.from_numpy(np.load(sig_path)).float())
                labels.append(self.class_to_idx[cls])

                row    = metadata.get((cls, fname), {})
                has_dt = float(str(row.get('has_dead_time', 'False')).lower() == 'true')
                dt_val = float(row.get('dead_time_value', 0.0)) / DT_VALUE_SCALE

                st  = float(row.get('settling_time',     0.0)) / REG_SCALES['settling_time']
                rt  = float(row.get('rise_time',         0.0)) / REG_SCALES['rise_time']
                ov  = float(row.get('overshoot_pct',     0.0)) / REG_SCALES['overshoot_pct']
                ssg = float(row.get('steady_state_gain', 0.0)) / REG_SCALES['steady_state_gain']

                has_dt_list.append(has_dt)
                dt_val_list.append(dt_val)
                reg_list.append([st, rt, ov, ssg])

        self.images      = torch.stack(images)
        self.signals     = torch.stack(signals)
        self.labels      = torch.tensor(labels,      dtype=torch.long)
        self.has_dt      = torch.tensor(has_dt_list, dtype=torch.float32)
        self.dt_vals     = torch.tensor(dt_val_list, dtype=torch.float32)
        self.reg_targets = torch.tensor(reg_list,    dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.images[idx],       # image tensor        (1, H, W)
            self.signals[idx],      # signal tensor       (2, 2000)
            self.labels[idx],       # class index         long
            self.has_dt[idx],       # has dead time       float 0/1
            self.dt_vals[idx],      # dead time value     float normalised
            self.reg_targets[idx],  # [settling, rise, overshoot, gain]  normalised
        )
