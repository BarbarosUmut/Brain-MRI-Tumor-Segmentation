import os
import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import albumentations as A  # Yeni kütüphane
from albumentations.pytorch import ToTensorV2

class LGGDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True):
        self.root_dir = root_dir
        self.mask_files = sorted(glob.glob(os.path.join(root_dir, '**/*_mask.tif'), recursive=True))
        self.image_files = [m.replace('_mask.tif', '.tif') for m in self.mask_files]
        self.image_files = [img for img in self.image_files if os.path.exists(img)]
        
        # Eğitim modu için güçlü artırma, doğrulama için sadece resize
        if train:
            self.transform = A.Compose([
                A.Resize(256, 256),
                A.HorizontalFlip(p=0.5), # %50 ihtimalle yatay çevir
                A.RandomRotate90(p=0.5), # %50 ihtimalle 90 derece döndür
                A.RandomBrightnessContrast(brightness_limit=0.2, # %20 oranında rastgele parlaklık değişimi
                                            contrast_limit=0.2,  # %20 oranında rastgele kontrast değişimi
                                            p=0.5),              # Bu işlemin uygulanma ihtimali %50), # Parlaklık değişimi
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3), # MRI'daki kumlanmayı simüle eder
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(256, 256),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        mask_path = self.mask_files[idx]
        
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 0).astype(np.float32) # Maskeyi 0-1 arasına çek

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].unsqueeze(0) # Kanal boyutu ekle (1, 256, 256)
        
        return image, mask