
import torch
import numpy as np
import cv2

def dice_coeff(pred, target, smooth=1.):
    """
    Calculate Dice Coefficient for a single batch.
    pred: predicted mask (logits or probabilities)
    target: ground truth mask
    """
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def calculate_tumor_area(mask_np):
    """
    Calculate tumor area in pixels.
    mask_np: numpy array of shape (H, W), values 0 or 1 (or 255)
    """
    # Ensure mask is binary
    mask_bin = (mask_np > 0.5).astype(np.uint8)
    area_pixels = np.sum(mask_bin)
    return float(area_pixels)

def calculate_tumor_location(mask_np):
    """
    Calculate the centroid of the tumor.
    Returns (cx, cy) or None if no tumor.
    """
    mask_bin = (mask_np > 0.5).astype(np.uint8)
    M = cv2.moments(mask_bin)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)
    return None

def overlay_mask(image_np, mask_np, color=(255, 0, 0), alpha=0.4):
    mask_bin = (mask_np > 0.5).astype(np.uint8)
    
    # EĞER MASKE TAMAMEN BOŞSA (Tümör yoksa)
    if np.sum(mask_bin) == 0:
        return image_np
        
    overlay = image_np.copy()
    color_mask = np.zeros_like(image_np)
    color_mask[mask_bin == 1] = color
    
    # Sadece maskenin olduğu piksellerde işlem yap
    mask_pixels = mask_bin == 1
    overlay[mask_pixels] = cv2.addWeighted(
        image_np[mask_pixels], 1 - alpha, 
        color_mask[mask_pixels], alpha, 0
    ).reshape(-1, 3) # Boyut hatasını önlemek için reshape
    
    return overlay