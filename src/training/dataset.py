import os
import glob
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


class SystemIDDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.img_dir = os.path.join(root_dir, 'images', split)
        self.sig_dir = os.path.join(root_dir, 'signals', split)

        if os.path.exists(self.img_dir):
            self.classes = sorted([f.name for f in os.scandir(self.img_dir) if f.is_dir()])
        else:
            self.classes = []

        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.transform = transforms.Compose([
            transforms.Grayscale(1),
            transforms.ToTensor()
        ])

        self.images = []
        self.signals = []
        self.labels = []

        # -------- PRELOAD EVERYTHING INTO RAM --------
        for cls in self.classes:
            img_paths = glob.glob(os.path.join(self.img_dir, cls, '*.png'))

            for img_path in img_paths:
                sig_path = os.path.join(
                    self.sig_dir,
                    cls,
                    os.path.basename(img_path).replace('.png', '.npy')
                )

                if not os.path.exists(sig_path):
                    continue

                with Image.open(img_path) as im:
                    img_tensor = self.transform(im)

                sig_array = np.load(sig_path)
                sig_tensor = torch.from_numpy(sig_array).float()

                self.images.append(img_tensor)
                self.signals.append(sig_tensor)
                self.labels.append(self.class_to_idx[cls])

        # Stack once for fast indexing
        self.images = torch.stack(self.images)
        self.signals = torch.stack(self.signals)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.images[idx],
            self.signals[idx],
            self.labels[idx]
        )