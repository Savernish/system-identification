import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.training.dataset import SystemIDDataset
from src.models.build_fusion_net import MultimodalFusionNet

# ── Loss weights ────────────────────────────────────────────────
W_CLASS  = 1.0   # transfer function class (main task)
W_DT_CLS = 0.5   # has dead time  (binary)
W_DT_VAL = 1.0   # dead time value (regression, masked to DT samples only)
W_REG    = 1.0   # settling time, rise time, overshoot, gain


def compute_loss(out, lbl, has_dt, dt_val, reg):
    ce  = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    loss_class  = ce(out['class_logits'], lbl)
    loss_dt_cls = bce(out['dt_logit'], has_dt)

    # Dead time value only penalised for samples that actually have dead time
    mask = has_dt.bool()
    if mask.any():
        loss_dt_val = mse(out['dt_value'][mask], dt_val[mask])
    else:
        loss_dt_val = torch.tensor(0.0, device=lbl.device)

    loss_reg = mse(out['reg'], reg)

    return (W_CLASS  * loss_class
          + W_DT_CLS * loss_dt_cls
          + W_DT_VAL * loss_dt_val
          + W_REG    * loss_reg)


CHECKPOINT_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', 'results', 'trained_nets', 'checkpoint.pth'
)


def save_checkpoint(epoch, model, optimizer, scheduler, best_val_loss, epochs_no_improve, best_weights):
    os.makedirs(os.path.dirname(os.path.abspath(CHECKPOINT_PATH)), exist_ok=True)
    torch.save({
        'epoch':             epoch,
        'model_state':       model.state_dict(),
        'optimizer_state':   optimizer.state_dict(),
        'scheduler_state':   scheduler.state_dict(),
        'best_val_loss':     best_val_loss,
        'epochs_no_improve': epochs_no_improve,
        'best_weights':      best_weights,
    }, CHECKPOINT_PATH)


def load_checkpoint(model, optimizer, scheduler):
    path = os.path.abspath(CHECKPOINT_PATH)
    if not os.path.exists(path):
        return 0, float('inf'), 0, None
    print(f'Resuming from checkpoint: {path}')
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    scheduler.load_state_dict(ckpt['scheduler_state'])
    return (
        ckpt['epoch'] + 1,
        ckpt['best_val_loss'],
        ckpt['epochs_no_improve'],
        ckpt['best_weights'],
    )


def train_fusion_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))

    print('Loading datasets...')
    train_dataset = SystemIDDataset(root_dir=root_dir, split='train')
    val_dataset   = SystemIDDataset(root_dir=root_dir, split='val')
    print(f'  Train: {len(train_dataset)}  Val: {len(val_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

    model     = MultimodalFusionNet(num_classes=6, num_reg_targets=4).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    epochs = 100
    patience = 15

    start_epoch, best_val_loss, epochs_no_improve, best_weights = load_checkpoint(
        model, optimizer, scheduler
    )

    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        # ── Train ──────────────────────────────────────────────
        model.train()
        t_loss = t_correct = t_total = 0

        for img, sig, lbl, has_dt, dt_val, reg in tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]', leave=False):
            img, sig = img.to(device), sig.to(device)
            lbl      = lbl.to(device)
            has_dt   = has_dt.to(device)
            dt_val   = dt_val.to(device)
            reg      = reg.to(device)

            optimizer.zero_grad()
            out  = model(img, sig)
            loss = compute_loss(out, lbl, has_dt, dt_val, reg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            t_loss    += loss.item() * img.size(0)
            t_correct += (out['class_logits'].argmax(1) == lbl).sum().item()
            t_total   += lbl.size(0)

        # ── Validation ─────────────────────────────────────────
        model.eval()
        v_loss = v_correct = v_total = 0
        dt_correct = dt_total = 0

        with torch.no_grad():
            for img, sig, lbl, has_dt, dt_val, reg in val_loader:
                img, sig = img.to(device), sig.to(device)
                lbl      = lbl.to(device)
                has_dt   = has_dt.to(device)
                dt_val   = dt_val.to(device)
                reg      = reg.to(device)

                out  = model(img, sig)
                loss = compute_loss(out, lbl, has_dt, dt_val, reg)

                v_loss    += loss.item() * img.size(0)
                v_correct += (out['class_logits'].argmax(1) == lbl).sum().item()
                v_total   += lbl.size(0)

                dt_pred    = (out['dt_logit'].sigmoid() > 0.5).float()
                dt_correct += (dt_pred == has_dt).sum().item()
                dt_total   += has_dt.size(0)

        avg_val_loss = v_loss / v_total
        cls_acc      = 100 * v_correct  / v_total
        dt_acc       = 100 * dt_correct / dt_total
        lr           = optimizer.param_groups[0]['lr']

        print(f'Epoch [{epoch+1:03d}/{epochs}]  {time.time()-t0:.1f}s  |  '
              f'T.Loss {t_loss/t_total:.4f}  V.Loss {avg_val_loss:.4f}  |  '
              f'Class {cls_acc:.1f}%  DT-detect {dt_acc:.1f}%  |  LR {lr:.2e}')

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss     = avg_val_loss
            best_weights      = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'\n[!] Early stopping. Best val loss: {best_val_loss:.4f}')
                save_checkpoint(epoch, model, optimizer, scheduler,
                                best_val_loss, epochs_no_improve, best_weights)
                break

        save_checkpoint(epoch, model, optimizer, scheduler,
                        best_val_loss, epochs_no_improve, best_weights)

    if best_weights:
        save_path = os.path.join(root_dir, '..', 'results', 'trained_nets', 'multitask_fusion_best.pth')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(best_weights, save_path)
        print(f'Saved: {save_path}')


if __name__ == '__main__':
    train_fusion_model()
