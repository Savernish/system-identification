import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.training.dataset import SystemIDDataset
from src.models.build_fusion_net import MultimodalFusionNet

def train_fusion_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
    train_dataset = SystemIDDataset(root_dir=root_dir, split='train')
    val_dataset = SystemIDDataset(root_dir=root_dir, split='val')

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

    model = MultimodalFusionNet(num_classes=6).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    
    # verbose=True parametresi kaldırıldı
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        patience=3, 
        factor=0.5,
        patience=3
    )

    epochs = 100
    patience_early_stop = 8
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_weights = None

    for epoch in range(epochs):
        start_time = time.time()
        
        # --- TRAIN ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for img, sig, label in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False):
            img, sig, label = img.to(device), sig.to(device), label.to(device)
            
            optimizer.zero_grad()
            out = model(img, sig)
            loss = criterion(out, label)
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item() * img.size(0)
            _, pred = torch.max(out, 1)
            train_total += label.size(0)
            train_correct += (pred == label).sum().item()

        # --- VALIDATION ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for img, sig, label in val_loader:
                img, sig, label = img.to(device), sig.to(device), label.to(device)
                out = model(img, sig)
                loss = criterion(out, label)
                
                val_loss += loss.item() * img.size(0)
                _, pred = torch.max(out, 1)
                val_total += label.size(0)
                val_correct += (pred == label).sum().item()

        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_acc = 100 * val_correct / val_total
        
        scheduler.step(val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{epochs}] | Süre: {time.time()-start_time:.1f}s | "
              f"T.Loss: {train_loss/len(train_dataset):.4f} | V.Loss: {epoch_val_loss:.4f} | "
              f"V.Acc: %{epoch_val_acc:.2f} | LR: {current_lr:.2e}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience_early_stop:
                print(f"\n[!] Early stopping devrede. En iyi Val Loss: {best_val_loss:.4f}")
                break

    if best_model_weights:
        save_path = os.path.join(root_dir, '..', 'results', 'trained_nets', 'grand_fusion_model_best.pth')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(best_model_weights, save_path)