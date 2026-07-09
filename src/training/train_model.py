import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.training.dataset import SystemIDDataset
from src.models.build_fusion_net import MultimodalFusionNet

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

    model     = MultimodalFusionNet(num_classes=6).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    epochs  = 100
    patience = 15

    start_epoch, best_val_loss, epochs_no_improve, best_weights = load_checkpoint(
        model, optimizer, scheduler
    )

    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        model.train()
        t_loss = t_correct = t_total = 0
        for img, sig, lbl, *_ in tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]', leave=False):
            img, sig, lbl = img.to(device), sig.to(device), lbl.to(device)

            optimizer.zero_grad()
            out  = model(img, sig)
            loss = criterion(out, lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            t_loss    += loss.item() * img.size(0)
            t_correct += (out.argmax(1) == lbl).sum().item()
            t_total   += lbl.size(0)

        model.eval()
        v_loss = v_correct = v_total = 0
        with torch.no_grad():
            for img, sig, lbl, *_ in val_loader:
                img, sig, lbl = img.to(device), sig.to(device), lbl.to(device)

                out  = model(img, sig)
                loss = criterion(out, lbl)

                v_loss    += loss.item() * img.size(0)
                v_correct += (out.argmax(1) == lbl).sum().item()
                v_total   += lbl.size(0)

        avg_val_loss = v_loss / v_total
        cls_acc      = 100 * v_correct / v_total
        lr           = optimizer.param_groups[0]['lr']

        print(f'Epoch [{epoch+1:03d}/{epochs}]  {time.time()-t0:.1f}s  |  '
              f'T.Loss {t_loss/t_total:.4f}  V.Loss {avg_val_loss:.4f}  |  '
              f'Acc {cls_acc:.1f}%  |  LR {lr:.2e}')

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
        save_path = os.path.join(root_dir, '..', 'results', 'trained_nets', 'fusion_best.pth')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(best_weights, save_path)
        print(f'Saved: {save_path}')


if __name__ == '__main__':
    train_fusion_model()
