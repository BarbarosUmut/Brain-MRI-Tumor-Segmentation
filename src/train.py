import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
from unet import UNet
from dataset import LGGDataset
from utils import dice_coeff

# --- KAYIP FONKSİYONU ---
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)       
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        return BCE + dice_loss

def train_model(data_dir, epochs=50, batch_size=16, learning_rate=1e-4, device='cpu'):
    print(f"--- Profesyonel Eğitim Başlatılıyor ---")
    print(f"Cihaz: {device} | Batch Size: {batch_size} | Hedef Epoch: {epochs}")
    
    # 1. Dataset ve Augmentation
    train_base_dataset = LGGDataset(data_dir, train=True)
    val_base_dataset = LGGDataset(data_dir, train=False)
    
    indices = list(range(len(train_base_dataset)))
    n_val = int(len(indices) * 0.1)
    n_train = len(indices) - n_val
    train_indices, val_indices = random_split(indices, [n_train, n_val])
    
    # Batch size burada kullanılıyor
    train_loader = DataLoader(
        Subset(train_base_dataset, train_indices), 
        batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        Subset(val_base_dataset, val_indices), 
        batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )
    
    print(f"Eğitim Görüntüsü: {n_train}, Doğrulama Görüntüsü: {n_val}")

    # 2. Model ve Optimizer (ADAM'a geçiş)
    model = UNet(n_channels=3, n_classes=1).to(device)
    
    # Adam genellikle U-Net ile daha hızlı yakınsar
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 3. SCHEDULER EKLENDİ (Kritik Nokta)
    # Eğer 'val_dice' 5 epoch boyunca artmazsa, öğrenme hızını 10'a böl.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.1, verbose=True)
    
    criterion = DiceBCELoss()
    best_dice = 0.0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs, true_masks = batch
                imgs = imgs.to(device)
                true_masks = true_masks.to(device)
                
                optimizer.zero_grad()
                masks_pred = model(imgs)
                
                loss = criterion(masks_pred, true_masks)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                pbar.update(imgs.shape[0])
                pbar.set_postfix(**{'loss': loss.item()})
            
        # Doğrulama
        val_score = evaluate(model, val_loader, device)
        
        # Scheduler'a skoru bildir: Eğer skor artmıyorsa LR'yi düşürecek
        scheduler.step(val_score)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Sonuç: Epoch {epoch+1} - Loss: {epoch_loss/len(train_loader):.4f} - Val Dice: {val_score:.4f} - LR: {current_lr}')
        
        # En iyi modeli kaydet
        if val_score > best_dice:
            best_dice = val_score
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/unet_best.pth')
            print(f"✅ REKOR! Yeni model kaydedildi. Dice: {val_score:.4f}")
        
        print("-" * 30)

    print(f"Eğitim bitti. En yüksek Dice Skoru: {best_dice:.4f}")

def evaluate(model, loader, device):
    model.eval()
    dice_score = 0
    n_steps = 0
    
    with torch.no_grad():
        for batch in loader:
            imgs, true_masks = batch
            imgs = imgs.to(device)
            true_masks = true_masks.to(device)
            
            mask_pred = model(imgs)
            dice_score += dice_coeff(mask_pred, true_masks).item()
            n_steps += 1
            
    return dice_score / n_steps

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/', help='Veri seti yolu')
    # Varsayılan epoch 50'ye, batch 8'e çıkarıldı
    parser.add_argument('--epochs', type=int, default=50, help='Epoch sayısı')
    parser.add_argument('--batch', type=int, default=16, help='Batch boyutu')
    parser.add_argument('--lr', type=float, default=1e-4, help='Öğrenme oranı')
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if os.path.exists(args.data):
        train_model(args.data, epochs=args.epochs, batch_size=args.batch, learning_rate=args.lr, device=device)
    else:
        print(f"Hata: '{args.data}' klasörü bulunamadı!")