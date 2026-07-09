"""
Evaluation for the fusion classification model.
Outputs: accuracy, per-class metrics, confusion matrices, confidence distribution.
Results saved to results/reports/evaluation/
"""

import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_recall_fscore_support,
)
from tabulate import tabulate

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.training.dataset import SystemIDDataset
from src.models.build_fusion_net import MultimodalFusionNet


def run_inference(model, loader, device):
    model.eval()
    preds, trues, confs = [], [], []

    with torch.no_grad():
        for img, sig, lbl, *_ in loader:
            img, sig = img.to(device), sig.to(device)
            out = model(img, sig)
            probs = torch.softmax(out, dim=1)
            conf, pred = probs.max(dim=1)
            preds.append(pred.cpu())
            trues.append(lbl)
            confs.append(conf.cpu())

    return (
        torch.cat(preds).numpy(),
        torch.cat(trues).numpy(),
        torch.cat(confs).numpy(),
    )


def plot_confusion_matrix(cm, class_names, title, path, normalised=False):
    data = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100 if normalised else cm
    fmt  = '.1f' if normalised else 'd'
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    suffix = ' (%)' if normalised else ''
    ax.set_title(f'{title}{suffix}', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Class')
    ax.set_xlabel('Predicted Class')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_confidence_distribution(confs, preds, trues, path):
    correct = preds == trues
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(confs[correct],  bins=40, alpha=0.7, label='Correct',   color='#2ecc71')
    ax.hist(confs[~correct], bins=40, alpha=0.7, label='Incorrect', color='#e74c3c')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Count')
    ax.set_title('Prediction Confidence Distribution', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()


def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n{"="*60}')
    print(f'  FUSION MODEL EVALUATION  |  Device: {device}')
    print(f'{"="*60}\n')

    data_dir = os.path.join(project_root, 'data')
    test_ds  = SystemIDDataset(root_dir=data_dir, split='test')
    loader   = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
    class_names = test_ds.classes
    print(f'Test set: {len(test_ds)} samples  |  {len(class_names)} classes\n')

    weights_path = os.path.join(project_root, 'results', 'trained_nets', 'fusion_best.pth')
    if not os.path.exists(weights_path):
        print(f'ERROR: weights not found at {weights_path}')
        return

    model = MultimodalFusionNet(num_classes=6).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))

    print('Running inference...')
    preds, trues, confs = run_inference(model, loader, device)

    acc = accuracy_score(trues, preds) * 100
    p_pc, rc_pc, f1_pc, sup_pc = precision_recall_fscore_support(
        trues, preds, average=None, zero_division=0
    )
    cm = confusion_matrix(trues, preds)

    print(f'{"─"*60}')
    print(f'  Overall accuracy: {acc:.2f}%\n')

    rows = []
    for i, cn in enumerate(class_names):
        rows.append([cn,
                     f'{p_pc[i]*100:.2f}%',
                     f'{rc_pc[i]*100:.2f}%',
                     f'{f1_pc[i]*100:.2f}%',
                     int(sup_pc[i]),
                     f'{cm[i,i]}/{int(sup_pc[i])}'])
    print(tabulate(rows,
                   headers=['Class', 'Precision', 'Recall', 'F1', 'Support', 'Correct/Total'],
                   tablefmt='fancy_grid', stralign='center'))

    print(f'\n{classification_report(trues, preds, target_names=class_names, digits=4)}')

    save_dir = os.path.join(project_root, 'results', 'reports', 'evaluation')
    os.makedirs(save_dir, exist_ok=True)

    plot_confusion_matrix(cm, class_names, 'Classification',
                          os.path.join(save_dir, 'confusion_matrix.png'))
    plot_confusion_matrix(cm, class_names, 'Classification (Normalised)',
                          os.path.join(save_dir, 'confusion_matrix_normalised.png'),
                          normalised=True)
    plot_confidence_distribution(confs, preds, trues,
                                 os.path.join(save_dir, 'confidence_distribution.png'))

    report_path = os.path.join(save_dir, 'report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('FUSION MODEL — EVALUATION REPORT\n\n')
        f.write(f'Test samples : {len(test_ds)}\n')
        f.write(f'Classes      : {class_names}\n\n')
        f.write(f'Accuracy     : {acc:.2f}%\n\n')
        f.write(classification_report(trues, preds, target_names=class_names, digits=4))

    print(f'\n  Plots + report saved to: {save_dir}')
    print(f'{"="*60}\n')


if __name__ == '__main__':
    evaluate()
