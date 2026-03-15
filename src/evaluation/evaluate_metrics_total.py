import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score,
    precision_recall_fscore_support
)
from tabulate import tabulate

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.training.dataset import SystemIDDataset
from src.models.build_fusion_net import MultimodalFusionNet
from src.models.build_fusion_net_lstm import MultimodalFusionNetLSTM
from src.models.build_image_only_net import ImageOnlyNet
from src.models.build_signal_only_net import SignalOnlyNet
from src.models.build_signal_only_lstm_net import SignalOnlyNetLSTM

# ──────────────────────────────────────────────────────────────
# Model registry — add new models here
# ──────────────────────────────────────────────────────────────
MODEL_CONFIG = {
    "fusion": {
        "class": MultimodalFusionNet,
        "weights": "grand_fusion_model_best.pth",
        "label": "Fusion (2D+1D CNN)",
        "input": "both",
    },
    "fusion_lstm": {
        "class": MultimodalFusionNetLSTM,
        "weights": "grand_fusion_model_lstm_best.pth",
        "label": "Fusion (2D+1D CNN+LSTM)",
        "input": "both",
    },
    "image_only": {
        "class": ImageOnlyNet,
        "weights": "image_only_model_best.pth",
        "label": "Image Only (2D CNN)",
        "input": "image",
    },
    "signal_only": {
        "class": SignalOnlyNet,
        "weights": "signal_only_model_best.pth",
        "label": "Signal Only (1D CNN)",
        "input": "signal",
    },
    "signal_only_lstm": {
        "class": SignalOnlyNetLSTM,
        "weights": "signal_only_model_lstm_best.pth",
        "label": "Signal Only (1D CNN+LSTM)",
        "input": "signal",
    },
}


# ──────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────
def run_inference(model, model_key, test_loader, device):
    """Run inference and return predictions, labels, and confidence scores."""
    cfg = MODEL_CONFIG[model_key]
    all_preds, all_labels, all_confs = [], [], []

    with torch.no_grad():
        for img_batch, sig_batch, labels in test_loader:
            if cfg["input"] == "both":
                img_batch, sig_batch = img_batch.to(device), sig_batch.to(device)
                outputs = model(img_batch, sig_batch)
            elif cfg["input"] == "image":
                img_batch = img_batch.to(device)
                outputs = model(img_batch)
            elif cfg["input"] == "signal":
                sig_batch = sig_batch.to(device)
                outputs = model(sig_batch)

            probs = F.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_confs.extend(confs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_confs)


# ──────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────
def plot_confusion_matrices(all_results, class_names, save_dir):
    """Plot all confusion matrices in a single figure."""
    n = len(all_results)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (model_key, data) in enumerate(all_results.items()):
        cm = data["cm"]
        label = MODEL_CONFIG[model_key]["label"]
        acc = data["accuracy"]

        ax = axes[idx]
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            ax=ax, cbar=False
        )
        ax.set_title(f'{label}\nAccuracy: {acc:.2f}%', fontsize=13, fontweight='bold')
        ax.set_ylabel('Gerçek Sınıf')
        ax.set_xlabel('Tahmin Edilen Sınıf')
        ax.tick_params(axis='x', rotation=45)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Karmaşıklık Matrisleri — Tüm Modeller (Test Seti)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, 'all_confusion_matrices.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  → Karmaşıklık matrisleri: {path}")


def plot_normalized_confusion_matrices(all_results, class_names, save_dir):
    """Plot normalized (percentage) confusion matrices."""
    n = len(all_results)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (model_key, data) in enumerate(all_results.items()):
        cm = data["cm"]
        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100
        label = MODEL_CONFIG[model_key]["label"]

        ax = axes[idx]
        sns.heatmap(
            cm_norm, annot=True, fmt='.1f', cmap='Oranges',
            xticklabels=class_names, yticklabels=class_names,
            ax=ax, cbar=False, vmin=0, vmax=100
        )
        ax.set_title(f'{label} (Yüzde %)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Gerçek Sınıf')
        ax.set_xlabel('Tahmin Edilen Sınıf')
        ax.tick_params(axis='x', rotation=45)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Normalize Karmaşıklık Matrisleri (Test Seti)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, 'all_confusion_matrices_normalized.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  → Normalize matrisler: {path}")


def plot_per_class_comparison(all_results, class_names, save_dir):
    """Bar chart comparing F1 score per class across all models."""
    model_keys = list(all_results.keys())
    model_labels = [MODEL_CONFIG[k]["label"] for k in model_keys]
    n_classes = len(class_names)
    n_models = len(model_keys)

    x = np.arange(n_classes)
    width = 0.8 / n_models

    fig, axes = plt.subplots(3, 1, figsize=(max(12, n_classes * 2), 18))
    metric_names = ['Precision', 'Recall', 'F1-Score']
    metric_keys = ['precision_per_class', 'recall_per_class', 'f1_per_class']
    cmaps = [plt.cm.Blues, plt.cm.Greens, plt.cm.Reds]

    for ax, metric_name, metric_key, cmap in zip(axes, metric_names, metric_keys, cmaps):
        for i, (mk, ml) in enumerate(zip(model_keys, model_labels)):
            values = all_results[mk][metric_key]
            color = cmap((i + 2) / (n_models + 3))
            bars = ax.bar(x + i * width, values * 100, width, label=ml, color=color, edgecolor='white')
            for bar, val in zip(bars, values):
                if bar.get_height() > 5:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 3,
                            f'{val*100:.0f}', ha='center', va='top', fontsize=7, fontweight='bold', color='white')

        ax.set_xlabel('Sınıf')
        ax.set_ylabel(f'{metric_name} (%)')
        ax.set_title(f'Sınıf Bazlı {metric_name} Karşılaştırması', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (n_models - 1) / 2)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_ylim(0, 110)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'per_class_comparison.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  → Sınıf bazlı karşılaştırma: {path}")


def plot_overall_comparison(all_results, save_dir):
    """Bar chart comparing overall accuracy, macro-F1, macro-precision, macro-recall."""
    model_keys = list(all_results.keys())
    model_labels = [MODEL_CONFIG[k]["label"] for k in model_keys]

    metrics = ['accuracy', 'macro_f1', 'macro_precision', 'macro_recall']
    metric_labels = ['Accuracy', 'Macro F1', 'Macro Precision', 'Macro Recall']

    x = np.arange(len(metrics))
    width = 0.8 / len(model_keys)
    colors = plt.cm.Set2(np.linspace(0, 1, len(model_keys)))

    fig, ax = plt.subplots(figsize=(12, 7))

    for i, (mk, ml) in enumerate(zip(model_keys, model_labels)):
        values = [all_results[mk][m] for m in metrics]
        bars = ax.bar(x + i * width, values, width, label=ml, color=colors[i], edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_ylabel('Skor (%)')
    ax.set_title('Genel Model Karşılaştırması (Test Seti)', fontsize=16, fontweight='bold')
    ax.set_xticks(x + width * (len(model_keys) - 1) / 2)
    ax.set_xticklabels(metric_labels, fontsize=12)
    ax.set_ylim(0, 105)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'overall_comparison.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  → Genel karşılaştırma: {path}")


def plot_confidence_distribution(all_results, save_dir):
    """Histogram of prediction confidence per model (correct vs incorrect)."""
    n = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (model_key, data) in zip(axes, all_results.items()):
        label = MODEL_CONFIG[model_key]["label"]
        confs = data["confidences"]
        preds = data["preds"]
        labels = data["labels"]

        correct_mask = preds == labels
        ax.hist(confs[correct_mask], bins=30, alpha=0.7, label='Doğru', color='#2ecc71', edgecolor='white')
        ax.hist(confs[~correct_mask], bins=30, alpha=0.7, label='Yanlış', color='#e74c3c', edgecolor='white')
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xlabel('Güven Skoru')
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)

    axes[0].set_ylabel('Örnek Sayısı')
    plt.suptitle('Güven Dağılımı (Doğru vs Yanlış Tahminler)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, 'confidence_distributions.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  → Güven dağılımı: {path}")


# ──────────────────────────────────────────────────────────────
# Main evaluation
# ──────────────────────────────────────────────────────────────
def evaluate_all():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'='*70}")
    print(f"  KAPSAMLI MODEL DEĞERLENDİRMESİ")
    print(f"  Cihaz: {device}")
    print(f"{'='*70}\n")

    data_dir = os.path.join(project_root, 'data')
    test_dataset = SystemIDDataset(root_dir=data_dir, split='test')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    class_names = test_dataset.classes
    n_classes = len(class_names)

    print(f"Test seti: {len(test_dataset)} örnek, {n_classes} sınıf")
    print(f"Sınıflar: {class_names}\n")

    all_results = {}
    available_models = []

    for model_key, cfg in MODEL_CONFIG.items():
        weights_path = os.path.join(project_root, 'results', 'trained_nets', cfg["weights"])
        if not os.path.exists(weights_path):
            print(f"  ⚠ {cfg['label']:30s} — ağırlık dosyası bulunamadı, atlanıyor.")
            continue

        print(f"  ▶ {cfg['label']:30s} değerlendiriliyor...", end=" ")

        model = cfg["class"](num_classes=n_classes).to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        model.eval()

        preds, labels, confs = run_inference(model, model_key, test_loader, device)

        acc = accuracy_score(labels, preds) * 100
        p_per_class, r_per_class, f1_per_class, support = precision_recall_fscore_support(
            labels, preds, average=None, zero_division=0
        )
        macro_p = precision_score(labels, preds, average='macro', zero_division=0) * 100
        macro_r = recall_score(labels, preds, average='macro', zero_division=0) * 100
        macro_f1 = f1_score(labels, preds, average='macro', zero_division=0) * 100
        weighted_f1 = f1_score(labels, preds, average='weighted', zero_division=0) * 100
        cm = confusion_matrix(labels, preds)

        all_results[model_key] = {
            "preds": preds,
            "labels": labels,
            "confidences": confs,
            "accuracy": acc,
            "macro_precision": macro_p,
            "macro_recall": macro_r,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "precision_per_class": p_per_class,
            "recall_per_class": r_per_class,
            "f1_per_class": f1_per_class,
            "support_per_class": support,
            "cm": cm,
            "mean_confidence": float(np.mean(confs)),
            "mean_conf_correct": float(np.mean(confs[preds == labels])) if (preds == labels).any() else 0,
            "mean_conf_wrong": float(np.mean(confs[preds != labels])) if (preds != labels).any() else 0,
        }
        available_models.append(model_key)
        print(f"Acc: {acc:.2f}%  |  F1: {macro_f1:.2f}%")

        del model
        torch.cuda.empty_cache()

    if not available_models:
        print("\n⛔ Hiçbir eğitilmiş model bulunamadı!")
        return

    print(f"\n{'='*70}")
    print(f"  DETAYLI SONUÇLAR")
    print(f"{'='*70}")

    print(f"\n{'─'*70}")
    print("  1) GENEL KARŞILAŞTIRMA TABLOSU")
    print(f"{'─'*70}")

    overview_headers = ["Model", "Accuracy", "Macro P", "Macro R", "Macro F1", "Weighted F1", "Avg Conf"]
    overview_rows = []
    for mk in available_models:
        d = all_results[mk]
        overview_rows.append([
            MODEL_CONFIG[mk]["label"],
            f"{d['accuracy']:.2f}%",
            f"{d['macro_precision']:.2f}%",
            f"{d['macro_recall']:.2f}%",
            f"{d['macro_f1']:.2f}%",
            f"{d['weighted_f1']:.2f}%",
            f"{d['mean_confidence']:.4f}",
        ])
    print(tabulate(overview_rows, headers=overview_headers, tablefmt="fancy_grid", stralign="center"))

    for mk in available_models:
        d = all_results[mk]
        label = MODEL_CONFIG[mk]["label"]

        print(f"\n{'─'*70}")
        print(f"  2) SINIF BAZLI METRİKLER — {label}")
        print(f"{'─'*70}")

        class_headers = ["Sınıf", "Precision", "Recall", "F1-Score", "Support", "Doğru/Toplam"]
        class_rows = []
        for i, cn in enumerate(class_names):
            correct = d["cm"][i][i]
            total = d["support_per_class"][i]
            class_rows.append([
                cn,
                f"{d['precision_per_class'][i]*100:.2f}%",
                f"{d['recall_per_class'][i]*100:.2f}%",
                f"{d['f1_per_class'][i]*100:.2f}%",
                int(total),
                f"{correct}/{int(total)}"
            ])
        print(tabulate(class_rows, headers=class_headers, tablefmt="fancy_grid", stralign="center"))

    print(f"\n{'─'*70}")
    print("  3) SINIF BAZLI F1-SCORE KARŞILAŞTIRMASI (Tüm Modeller)")
    print(f"{'─'*70}")

    f1_headers = ["Sınıf"] + [MODEL_CONFIG[mk]["label"] for mk in available_models]
    f1_rows = []
    for i, cn in enumerate(class_names):
        row = [cn]
        f1_values = []
        for mk in available_models:
            val = all_results[mk]["f1_per_class"][i] * 100
            f1_values.append(val)
        best_val = max(f1_values)
        for val in f1_values:
            if val == best_val:
                row.append(f"★ {val:.2f}%")
            else:
                row.append(f"  {val:.2f}%")
        f1_rows.append(row)
    print(tabulate(f1_rows, headers=f1_headers, tablefmt="fancy_grid", stralign="center"))

    print(f"\n{'─'*70}")
    print("  4) GÜVEN ANALİZİ")
    print(f"{'─'*70}")

    conf_headers = ["Model", "Ortalama Güven", "Doğru Tahmin Güveni", "Yanlış Tahmin Güveni"]
    conf_rows = []
    for mk in available_models:
        d = all_results[mk]
        conf_rows.append([
            MODEL_CONFIG[mk]["label"],
            f"{d['mean_confidence']:.4f}",
            f"{d['mean_conf_correct']:.4f}",
            f"{d['mean_conf_wrong']:.4f}",
        ])
    print(tabulate(conf_rows, headers=conf_headers, tablefmt="fancy_grid", stralign="center"))

    for mk in available_models:
        d = all_results[mk]
        label = MODEL_CONFIG[mk]["label"]
        print(f"\n{'─'*70}")
        print(f"  5) SKLEARN RAPORU — {label}")
        print(f"{'─'*70}")
        print(classification_report(d["labels"], d["preds"], target_names=class_names, digits=4))

    report_dir = os.path.join(project_root, 'results', 'reports', 'total_evaluation')
    os.makedirs(report_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  GRAFİKLER OLUŞTURULUYOR")
    print(f"{'='*70}")

    plot_confusion_matrices(all_results, class_names, report_dir)
    plot_normalized_confusion_matrices(all_results, class_names, report_dir)
    plot_per_class_comparison(all_results, class_names, report_dir)
    plot_overall_comparison(all_results, report_dir)
    plot_confidence_distribution(all_results, report_dir)

    report_path = os.path.join(report_dir, 'evaluation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("KAPSAMLI MODEL DEĞERLENDİRME RAPORU\n")
        f.write(f"Test Seti: {len(test_dataset)} örnek, {n_classes} sınıf\n")
        f.write(f"Sınıflar: {class_names}\n\n")

        f.write("GENEL KARŞILAŞTIRMA\n")
        f.write(tabulate(overview_rows, headers=overview_headers, tablefmt="fancy_grid", stralign="center"))
        f.write("\n\n")

        f.write("SINIF BAZLI F1-SCORE KARŞILAŞTIRMASI\n")
        f.write(tabulate(f1_rows, headers=f1_headers, tablefmt="fancy_grid", stralign="center"))
        f.write("\n\n")

        for mk in available_models:
            d = all_results[mk]
            f.write(f"\n{'='*50}\n{MODEL_CONFIG[mk]['label']}\n{'='*50}\n")
            f.write(classification_report(d["labels"], d["preds"], target_names=class_names, digits=4))
            f.write("\n")

    print(f"  → Metin rapor: {report_path}")

    print(f"\n{'='*70}")
    print(f"  DEĞERLENDİRME TAMAMLANDI")
    print(f"  Tüm görseller: {report_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    evaluate_all()
