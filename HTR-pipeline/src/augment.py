import cv2
import os
import numpy as np
import random

def augment_image(img):
    augmentations = []
    
    # 1. Original
    augmentations.append(img)
    
    # 2. Rotations (small angles)
    rows, cols = img.shape
    for angle in [-10, -5, 5, 10]:
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        rotated = cv2.warpAffine(img, M, (cols, rows))
        augmentations.append(rotated)
        
    # 3. Brightness (luminosity)
    bright = cv2.convertScaleAbs(img, alpha=1.2, beta=30)
    augmentations.append(bright)
    
    # 4. Erosion/Dilation (simulates thin or thick pens)
    kernel = np.ones((2,2), np.uint8)
    dilated = cv2.dilate(img, kernel, iterations=1)
    augmentations.append(dilated)
    
    return augmentations

def process_dataset(base_path):
    for label in os.listdir(base_path):
        label_dir = os.path.join(base_path, label)
        if not os.path.isdir(label_dir): continue
        
        for img_name in os.listdir(label_dir):
            if not img_name.endswith('.png'): continue
            
            img_path = os.path.join(label_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None: continue
            
            # Create variations
            variations = augment_image(img)
            
            # Save variations
            for i, v in enumerate(variations):
                new_name = f"aug_{i}_{img_name}"
                cv2.imwrite(os.path.join(label_dir, new_name), v)