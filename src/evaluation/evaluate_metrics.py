import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.training.dataset import SystemIDDataset
from src.models.build_fusion_net import MultimodalFusionNet
from src.models.build_fusion_net_lstm import MultimodalFusionNetLSTM
from src.models.build_image_only_net import ImageOnlyNet
from src.models.build_signal_only_net import SignalOnlyNet
from src.models.build_signal_only_lstm_net import SignalOnlyNetLSTM


MODEL_CONFIG = {
    "fusion": {
        "class": MultimodalFusionNet,
        "weights": "grand_fusion_model_best.pth",
        "label": "Fusion (2D+1D CNN)",
    },
    "fusion_lstm": {
        "class": MultimodalFusionNetLSTM,
        "weights": "grand_fusion_model_lstm_best.pth",
        "label": "Fusion (2D+1D CNN+LSTM)",
    },
    "image_only": {
        "class": ImageOnlyNet,
        "weights": "image_only_model_best.pth",
        "label": "Image Only (2D CNN)",
    },
    "signal_only": {
        "class": SignalOnlyNet,
        "weights": "signal_only_model_best.pth",
        "label": "Signal Only (1D CNN)",
    },
    "signal_only_lstm": {
        "class": SignalOnlyNetLSTM,
        "weights": "signal_only_model_lstm_best.pth",
        "label": "Signal Only (1D CNN + LSTM)",
    },
}


def evaluate_model(model_type="fusion"):
    cfg = MODEL_CONFIG[model_type]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Değerlendirme başlatılıyor... Model: {cfg['label']} | Cihaz: {device}")

    data_dir = os.path.join(project_root, 'data')
    test_dataset = SystemIDDataset(root_dir=data_dir, split='test')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    model = cfg["class"](num_classes=6).to(device)
    model_path = os.path.join(project_root, 'results', 'trained_nets', cfg["weights"])

    if not os.path.exists(model_path):
        print(f"Hata: Eğitilmiş model bulunamadı ({model_path}).")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    print("Test verisi işleniyor...")
    with torch.no_grad():
        for img_batch, sig_batch, labels in test_loader:
            if model_type == "fusion":
                img_batch, sig_batch = img_batch.to(device), sig_batch.to(device)
                outputs = model(img_batch, sig_batch)
            elif model_type == "fusion_lstm":
                img_batch, sig_batch = img_batch.to(device), sig_batch.to(device)
                outputs = model(img_batch, sig_batch)
            elif model_type == "image_only":
                img_batch = img_batch.to(device)
                outputs = model(img_batch)
            elif model_type == "signal_only":
                sig_batch = sig_batch.to(device)
                outputs = model(sig_batch)
            elif model_type == "signal_only_lstm":
                sig_batch = sig_batch.to(device)
                outputs = model(sig_batch)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    class_names = test_dataset.classes
    print(f"\n--- Sınıflandırma Raporu: {cfg['label']} ---")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Karmaşıklık Matrisi - {cfg["label"]} (Test Seti)')
    plt.ylabel('Gerçek Sınıf')
    plt.xlabel('Tahmin Edilen Sınıf')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    report_dir = os.path.join(project_root, 'results', 'reports')
    os.makedirs(report_dir, exist_ok=True)
    cm_path = os.path.join(report_dir, f'confusion_matrix_{model_type}.png')

    plt.savefig(cm_path, dpi=300)
    print(f"\nKarmaşıklık matrisi kaydedildi: {cm_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Değerlendirme")
    parser.add_argument(
        "--model",
        type=str,
        default="fusion",
        choices=["fusion", "fusion_lstm", "image_only", "signal_only", "signal_only_lstm"],
        help="Değerlendirilecek model tipi (default: fusion)"
    )
    args = parser.parse_args()
    evaluate_model(args.model)