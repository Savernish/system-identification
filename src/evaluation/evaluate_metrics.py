# src/evaluation/evaluate_metrics.py
"""
Comprehensive evaluation for the multi-task fusion model.

Sections:
  1. Classification  — accuracy, confusion matrix, per-class P/R/F1
  2. Dead Time       — binary detection accuracy, P/R/F1, CM
  3. Regression      — MAE & RMSE for each regression head (denormalised)

Outputs saved to results/reports/evaluation/
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

from src.training.dataset import SystemIDDataset, REG_SCALES, DT_VALUE_SCALE
from src.models.build_fusion_net import MultimodalFusionNet

REG_NAMES   = ['Dead Time Value', 'Settling Time', 'Rise Time', 'Overshoot %', 'Steady-State Gain']
REG_UNITS   = ['s', 's', 's', '%', '']
# Denorm scales in same order as dataset returns (dt_val separate, then reg[4])
REG_DENORM  = [DT_VALUE_SCALE,
               REG_SCALES['settling_time'],
               REG_SCALES['rise_time'],
               REG_SCALES['overshoot_pct'],
               REG_SCALES['steady_state_gain']]


# ──────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────

def run_inference(model, loader, device):
    model.eval()
    cls_preds, cls_true = [], []
    dt_preds,  dt_true  = [], []
    dt_val_preds, dt_val_true = [], []
    reg_preds, reg_true = [], []
    cls_confs = []

    with torch.no_grad():
        for img, sig, lbl, has_dt, dt_val, reg in loader:
            img, sig = img.to(device), sig.to(device)
            out = model(img, sig)

            probs     = torch.softmax(out['class_logits'], dim=1)
            conf, pred = probs.max(dim=1)

            cls_preds.append(pred.cpu())
            cls_true.append(lbl)
            cls_confs.append(conf.cpu())

            dt_bin = (out['dt_logit'].sigmoid() > 0.5).float().cpu()
            dt_preds.append(dt_bin)
            dt_true.append(has_dt)

            dt_val_preds.append(out['dt_value'].cpu())
            dt_val_true.append(dt_val)

            reg_preds.append(out['reg'].cpu())
            reg_true.append(reg)

    return {
        'cls_preds':    torch.cat(cls_preds).numpy(),
        'cls_true':     torch.cat(cls_true).numpy(),
        'cls_confs':    torch.cat(cls_confs).numpy(),
        'dt_preds':     torch.cat(dt_preds).numpy(),
        'dt_true':      torch.cat(dt_true).numpy(),
        'dt_val_preds': torch.cat(dt_val_preds).numpy(),
        'dt_val_true':  torch.cat(dt_val_true).numpy(),
        'reg_preds':    torch.cat(reg_preds).numpy(),   # (N, 4)
        'reg_true':     torch.cat(reg_true).numpy(),    # (N, 4)
    }


# ──────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm, class_names, title, path, normalised=False):
    data   = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100 if normalised else cm
    fmt    = '.1f' if normalised else 'd'
    suffix = ' (%)' if normalised else ''
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar=True)
    ax.set_title(f'{title}{suffix}', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Class')
    ax.set_xlabel('Predicted Class')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_dt_confusion_matrix(dt_true, dt_preds, path):
    cm = confusion_matrix(dt_true, dt_preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=['No DT', 'Has DT'],
                yticklabels=['No DT', 'Has DT'], ax=ax)
    ax.set_title('Dead Time Detection', fontsize=13, fontweight='bold')
    ax.set_ylabel('True')
    ax.set_xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_regression_scatter(r, save_dir):
    """5 scatter plots: predicted vs true for each regression target."""
    # Build combined arrays: [dt_val, settling, rise, overshoot, gain]
    all_true  = np.column_stack([r['dt_val_true'],  r['reg_true']])   # (N, 5)
    all_preds = np.column_stack([r['dt_val_preds'], r['reg_preds']])   # (N, 5)

    # Denormalise
    for i, scale in enumerate(REG_DENORM):
        all_true[:, i]  *= scale
        all_preds[:, i] *= scale

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, (name, unit, scale) in enumerate(zip(REG_NAMES, REG_UNITS, REG_DENORM)):
        ax  = axes[i]
        t   = all_true[:, i]
        p   = all_preds[:, i]

        # For dead time value: only plot samples that actually have dead time
        if i == 0:
            mask = r['dt_true'] > 0.5
            t, p = t[mask], p[mask]
            name = 'Dead Time Value\n(DT samples only)'

        ax.scatter(t, p, alpha=0.3, s=8, color='steelblue')
        lim = max(t.max(), p.max()) * 1.05
        ax.plot([0, lim], [0, lim], 'r--', linewidth=1.2, label='Perfect')
        mae  = np.mean(np.abs(p - t))
        rmse = np.sqrt(np.mean((p - t) ** 2))
        ax.set_title(f'{name}\nMAE={mae:.3f}{unit}  RMSE={rmse:.3f}{unit}', fontsize=10)
        ax.set_xlabel(f'True ({unit})')
        ax.set_ylabel(f'Predicted ({unit})')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    axes[5].set_visible(False)
    plt.suptitle('Regression: Predicted vs True', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'regression_scatter.png'), dpi=200,
                bbox_inches='tight', facecolor='white')
    plt.close()


def plot_confidence_distribution(cls_confs, cls_preds, cls_true, path):
    correct = cls_preds == cls_true
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(cls_confs[correct],  bins=40, alpha=0.7, label='Correct',   color='#2ecc71')
    ax.hist(cls_confs[~correct], bins=40, alpha=0.7, label='Incorrect', color='#e74c3c')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Count')
    ax.set_title('Prediction Confidence Distribution', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n{"="*65}')
    print(f'  MULTI-TASK MODEL EVALUATION')
    print(f'  Device: {device}')
    print(f'{"="*65}\n')

    data_dir = os.path.join(project_root, 'data')
    print('Loading test set...')
    test_ds     = SystemIDDataset(root_dir=data_dir, split='test')
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
    class_names = test_ds.classes
    print(f'  {len(test_ds)} samples  |  {len(class_names)} classes\n')

    weights_path = os.path.join(project_root, 'results', 'trained_nets', 'multitask_fusion_best.pth')
    if not os.path.exists(weights_path):
        print(f'ERROR: weights not found at {weights_path}')
        return

    model = MultimodalFusionNet(num_classes=6, num_reg_targets=4).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    print('Running inference...')
    r = run_inference(model, test_loader, device)

    save_dir = os.path.join(project_root, 'results', 'reports', 'evaluation')
    os.makedirs(save_dir, exist_ok=True)

    # ── 1. Classification ──────────────────────────────────────
    cls_acc = accuracy_score(r['cls_true'], r['cls_preds']) * 100
    p_pc, rc_pc, f1_pc, sup_pc = precision_recall_fscore_support(
        r['cls_true'], r['cls_preds'], average=None, zero_division=0
    )
    cm = confusion_matrix(r['cls_true'], r['cls_preds'])

    print(f'\n{"─"*65}')
    print('  1. CLASSIFICATION')
    print(f'{"─"*65}')
    print(f'  Overall accuracy: {cls_acc:.2f}%\n')

    cls_rows = []
    for i, cn in enumerate(class_names):
        cls_rows.append([cn,
                         f'{p_pc[i]*100:.2f}%',
                         f'{rc_pc[i]*100:.2f}%',
                         f'{f1_pc[i]*100:.2f}%',
                         int(sup_pc[i]),
                         f'{cm[i,i]}/{int(sup_pc[i])}'])
    print(tabulate(cls_rows,
                   headers=['Class', 'Precision', 'Recall', 'F1', 'Support', 'Correct/Total'],
                   tablefmt='fancy_grid', stralign='center'))

    print(f'\n  Sklearn report:\n')
    print(classification_report(r['cls_true'], r['cls_preds'],
                                target_names=class_names, digits=4))

    # ── 2. Dead Time Detection ─────────────────────────────────
    dt_acc = accuracy_score(r['dt_true'], r['dt_preds']) * 100
    dt_p, dt_r, dt_f1, _ = precision_recall_fscore_support(
        r['dt_true'], r['dt_preds'], average='binary', zero_division=0
    )

    print(f'{"─"*65}')
    print('  2. DEAD TIME DETECTION')
    print(f'{"─"*65}')
    dt_rows = [['Accuracy', f'{dt_acc:.2f}%'],
               ['Precision', f'{dt_p*100:.2f}%'],
               ['Recall',    f'{dt_r*100:.2f}%'],
               ['F1',        f'{dt_f1*100:.2f}%']]
    print(tabulate(dt_rows, tablefmt='fancy_grid', stralign='center'))

    # ── 3. Regression ──────────────────────────────────────────
    print(f'\n{"─"*65}')
    print('  3. REGRESSION  (denormalised)')
    print(f'{"─"*65}')

    all_true  = np.column_stack([r['dt_val_true'],  r['reg_true']])
    all_preds = np.column_stack([r['dt_val_preds'], r['reg_preds']])
    for i, scale in enumerate(REG_DENORM):
        all_true[:, i]  *= scale
        all_preds[:, i] *= scale

    reg_rows = []
    for i, (name, unit) in enumerate(zip(REG_NAMES, REG_UNITS)):
        t, p = all_true[:, i], all_preds[:, i]
        if i == 0:                        # DT value: mask to DT-only samples
            mask = r['dt_true'] > 0.5
            t, p = t[mask], p[mask]
        mae  = np.mean(np.abs(p - t))
        rmse = np.sqrt(np.mean((p - t) ** 2))
        reg_rows.append([f'{name} ({unit})' if unit else name,
                         f'{mae:.4f}', f'{rmse:.4f}'])

    print(tabulate(reg_rows, headers=['Target', 'MAE', 'RMSE'],
                   tablefmt='fancy_grid', stralign='center'))

    # ── Plots ──────────────────────────────────────────────────
    print(f'\n{"─"*65}')
    print('  Saving plots...')
    plot_confusion_matrix(cm, class_names, 'Class Classification',
                          os.path.join(save_dir, 'confusion_matrix.png'))
    plot_confusion_matrix(cm, class_names, 'Class Classification (Normalised)',
                          os.path.join(save_dir, 'confusion_matrix_normalised.png'),
                          normalised=True)
    plot_dt_confusion_matrix(r['dt_true'], r['dt_preds'],
                             os.path.join(save_dir, 'dead_time_detection_cm.png'))
    plot_regression_scatter(r, save_dir)
    plot_confidence_distribution(r['cls_confs'], r['cls_preds'], r['cls_true'],
                                 os.path.join(save_dir, 'confidence_distribution.png'))

    # ── Text report ────────────────────────────────────────────
    report_path = os.path.join(save_dir, 'report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('MULTI-TASK FUSION MODEL — EVALUATION REPORT\n\n')
        f.write(f'Test samples : {len(test_ds)}\n')
        f.write(f'Classes      : {class_names}\n\n')
        f.write(f'Classification accuracy : {cls_acc:.2f}%\n')
        f.write(f'Dead time detection acc : {dt_acc:.2f}%\n\n')
        f.write('CLASSIFICATION\n')
        f.write(classification_report(r['cls_true'], r['cls_preds'],
                                      target_names=class_names, digits=4))
        f.write('\n\nREGRESSION\n')
        f.write(tabulate(reg_rows, headers=['Target', 'MAE', 'RMSE'],
                         tablefmt='fancy_grid', stralign='center'))

    print(f'\n  All files saved to: {save_dir}')
    print(f'{"="*65}\n')


if __name__ == '__main__':
    evaluate()
