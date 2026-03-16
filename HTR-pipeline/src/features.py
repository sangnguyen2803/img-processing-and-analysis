import cv2
import numpy as np
from skimage.feature import hog
import matplotlib.pyplot as plt
from skimage import exposure

def extract_hu(img, visualize=False):
    moments = cv2.moments(img)
    hu = cv2.HuMoments(moments).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    
    if visualize:
        plt.figure(figsize=(5,3))
        colors = plt.cm.viridis((hu_log - hu_log.min()) / (hu_log.max() - hu_log.min()))
        plt.bar(np.arange(1,8), hu_log, color=colors)
        plt.xlabel("Hu Moment Index")
        plt.ylabel("Magnitude (log scale)")
        plt.title("Hu Moments")
        plt.show()
    
    return hu_log

def extract_hog(img, visualize=False):
    hog_feat = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )
    
    if visualize:
        hog_image = hog(
            img,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            visualize=True
        )[1]
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        plt.figure(figsize=(4,4))
        plt.imshow(hog_image_rescaled, cmap='gray')
        plt.title("HOG Visualization")
        plt.axis('off')
        plt.show()
    
    return hog_feat

def extract_features(img, visualize=False):
    hu_feat = extract_hu(img, visualize=visualize)
    hog_feat = extract_hog(img, visualize=visualize)
    return np.hstack([hog_feat, hu_feat])

def extract_geometric(img):
    # Find all external contours (the letter should be the largest one)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Return a zero vector if no contour is found
    if not contours:
        return np.zeros(3)
    
    # Select the largest contour (assuming this is the character)
    cnt = max(contours, key=cv2.contourArea)
    # Bounding box around the character
    x, y, w, h = cv2.boundingRect(cnt)
    
    # 1. Aspect Aspect ratio (width / height)
    aspect_ratio = float(w)/h
    
    # 2. ratio of contour area to bounding box area
    area = cv2.contourArea(cnt)
    rect_area = w*h
    extent = float(area)/rect_area if rect_area != 0 else 0
    
    # 3. Solidity - ratio of contour area to convex hull area
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area)/hull_area if hull_area != 0 else 0
    
    return np.array([aspect_ratio, extent, solidity])