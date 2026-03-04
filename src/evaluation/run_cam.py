# src/evaluation/run_cam.py
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.training.dataset import SystemIDDataset
from src.models.build_fusion_net import MultimodalFusionNet

class MultimodalWrapper(torch.nn.Module):
    def __init__(self, model, sig_input):
        super().__init__()
        self.model = model
        self.sig_input = sig_input
    def forward(self, x):
        return self.model(x, self.sig_input)

def run_multiclass_cam():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalFusionNet(num_classes=6).to(device)
    model_path = os.path.join(project_root, 'results', 'trained_nets', 'grand_fusion_model_best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_dataset = SystemIDDataset(root_dir=os.path.join(project_root, 'data'), split='test')
    class_names = test_dataset.classes
    num_classes = len(class_names)

    selected_indices = []
    found_classes = set()
    for i in range(len(test_dataset)):
        _, _, label = test_dataset[i]
        if label.item() not in found_classes:
            selected_indices.append(i)
            found_classes.add(label.item())
        if len(selected_indices) == num_classes: break

    # Grafik Ayarları (Beyaz Arka Plan)
    plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white'})
    fig, axes = plt.subplots(num_classes, 2, figsize=(14, 3 * num_classes))
    target_layers = [model.branch_2d[8]]

    for idx, sample_idx in enumerate(selected_indices):
        img_tensor, sig_tensor, label = test_dataset[sample_idx]
        img_input = img_tensor.unsqueeze(0).to(device)
        sig_input = sig_tensor.unsqueeze(0).to(device)

        wrapper_model = MultimodalWrapper(model, sig_input)
        cam = GradCAM(model=wrapper_model, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=img_input, targets=None)[0, :]

        with torch.no_grad():
            output = model(img_input, sig_input)
            probs = F.softmax(output, dim=1)
            conf, pred = torch.max(probs, dim=1)
            conf_val = conf.item() * 100
            pred_class = class_names[pred.item()]

        img_np = img_tensor.squeeze().cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        img_rgb = np.repeat(img_np[:, :, np.newaxis], 3, axis=2)
        visualization = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True)

        true_class = class_names[label.item()]
        
        # Orijinal Görüntü
        axes[idx, 0].imshow(img_np, cmap='gray')
        axes[idx, 0].set_title(f"Target: {true_class}", fontsize=14, color='black', fontweight='bold')
        axes[idx, 0].axis('off')

        # CAM Görüntüsü
        axes[idx, 1].imshow(visualization)
        color = 'green' if true_class == pred_class else 'red'
        axes[idx, 1].set_title(f"Pred: {pred_class} ({conf_val:.1f}%)", fontsize=14, color=color, fontweight='bold')
        axes[idx, 1].axis('off')

    plt.subplots_adjust(hspace=0.4, wspace=0.1)
    report_path = os.path.join(project_root, 'results', 'reports', 'multiclass_gradcam_white.png')
    plt.savefig(report_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Rapor kaydedildi: {report_path}")
    plt.show()

if __name__ == "__main__":
    run_multiclass_cam()