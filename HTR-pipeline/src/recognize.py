import os
import cv2
import numpy as np
import shutil
from sklearn.metrics import classification_report
from joblib import load
from features import extract_features
from segmentation import preprocess, resize_and_pad

def extract_characters(image_path, min_area=50, debug_dir="segmented_chars"):
    if os.path.exists(debug_dir):
        shutil.rmtree(debug_dir)
    os.makedirs(debug_dir, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None: raise FileNotFoundError(f"Cannot read {image_path}")

    bin_img = preprocess(img)
    char_count = 0
    all_chars = []

    # 1. Line segmentation
    row_sum = np.sum(bin_img, axis=1)
    thresh = 10
    line_indices = []
    in_line = False
    start_row = 0
    for i, val in enumerate(row_sum):
        if val > thresh and not in_line:
            in_line = True
            start_row = i
        elif val <= thresh and in_line:
            in_line = False
            line_indices.append((start_row, i))
    if in_line: line_indices.append((start_row, len(row_sum)))

    for line_idx, (y1, y2) in enumerate(line_indices):
        line_img = bin_img[y1:y2, :]
        
        # Use erosion to help separate touching letters
        kernel = np.ones((2,2), np.uint8)
        eroded_line = cv2.erode(line_img, kernel, iterations=1)
        contours, _ = cv2.findContours(eroded_line, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get initial bounding boxes
        rects = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h > min_area:
                rects.append([x, y, w, h])

        if not rects: continue

        rects = sorted(rects, key=lambda r: r[0])
        merged_rects = []
        used = set()

        for i in range(len(rects)):
            if i in used: continue
            curr_x, curr_y, curr_w, curr_h = rects[i]
            
            for j in range(len(rects)):
                if i == j or j in used: continue
                next_x, next_y, next_w, next_h = rects[j]

                x_overlap = not (next_x > curr_x + curr_w + 2 or curr_x > next_x + next_w + 2)
                
                if x_overlap:
                    new_x = min(curr_x, next_x)
                    new_y = min(curr_y, next_y)
                    new_w = max(curr_x + curr_w, next_x + next_w) - new_x
                    new_h = max(curr_y + curr_h, next_y + next_h) - new_y
                    
                    curr_x, curr_y, curr_w, curr_h = new_x, new_y, new_w, new_h
                    used.add(j)

            merged_rects.append([curr_x, curr_y, curr_w, curr_h])
            used.add(i)

        # Sort merged rects from left to right
        merged_rects = sorted(merged_rects, key=lambda r: r[0])

        # 2. Crop and Save
        for x, y, w, h in merged_rects:
            # Re-crop from ORIGINAL line_img
            cimg = line_img[y:y+h, x:x+w]

            # Split wide blocks (m, w, or connected letters)
            if w > 1.7 * h:
                half_w = w // 2
                crops = [cimg[:, :half_w], cimg[:, half_w:]]
            else:
                crops = [cimg]

            for final_crop in crops:
                char_count += 1
                processed_char = resize_and_pad(final_crop)
                filename = f"L{line_idx:02d}_C{char_count:03d}.png"
                cv2.imwrite(os.path.join(debug_dir, filename), processed_char)
                all_chars.append(processed_char)

    return all_chars
# -----------------------------
# Recognize text with SVM and RF
# -----------------------------
def recognize_text(image_path):
    svm = load("models/svm_model.joblib")
    rf = load("models/rf_model.joblib")
    scaler = load("models/svm_scaler.joblib")

    chars = extract_characters(image_path)

    svm_preds, rf_preds = [], []

    for char in chars:
        feat = extract_features(char)
        if feat is None:
            continue
  
        svm_preds.append(svm.predict(scaler.transform([feat]))[0])
        rf_preds.append(rf.predict([feat])[0])

    return svm_preds, rf_preds

# -----------------------------
# This evalutation needs more work
# -----------------------------
def evaluate(y_true_str, y_pred_list):
    # Convert ground truth string to list of characters
    y_true = list(y_true_str)
    
    # Trim to the shorter of the two to avoid crash
    min_len = min(len(y_true), len(y_pred_list))
    y_true = y_true[:min_len]
    y_pred = y_pred_list[:min_len]
    

    print(classification_report(y_true, y_pred, zero_division=0))

