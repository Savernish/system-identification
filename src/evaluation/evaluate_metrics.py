# src/evaluation/evaluate_metrics.py
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.training.dataset import SystemIDDataset
from src.models.build_fusion_net import MultimodalFusionNet

def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Değerlendirme başlatılıyor... Cihaz: {device}")

    # Test verisini yükle
    data_dir = os.path.join(project_root, 'data')
    test_dataset = SystemIDDataset(root_dir=data_dir, split='test')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Modeli yükle
    model = MultimodalFusionNet(num_classes=6).to(device)
    model_path = os.path.join(project_root, 'results', 'trained_nets', 'grand_fusion_model_best.pth')
    
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
            img_batch, sig_batch = img_batch.to(device), sig_batch.to(device)
            outputs = model(img_batch, sig_batch)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Sınıflandırma Raporu
    class_names = test_dataset.classes
    print("\n--- Sınıflandırma Raporu (Classification Report) ---")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Karmaşıklık Matrisi (Confusion Matrix)
    cm = confusion_matrix(all_labels, all_preds)
    
    # Görselleştirme ve Kaydetme
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Karmaşıklık Matrisi (Test Seti)')
    plt.ylabel('Gerçek Sınıf')
    plt.xlabel('Tahmin Edilen Sınıf')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    report_dir = os.path.join(project_root, 'results', 'reports')
    os.makedirs(report_dir, exist_ok=True)
    cm_path = os.path.join(report_dir, 'confusion_matrix.png')
    
    plt.savefig(cm_path, dpi=300)
    print(f"\nKarmaşıklık matrisi kaydedildi: {cm_path}")

if __name__ == "__main__":
    evaluate_model()