import cv2
import os
import numpy as np
import string

IMG_SIZE = 32

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    _, bin_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bin_img

def resize_and_pad(img):
    if img is None or img.size == 0:
        return None
    h, w = img.shape
    scale = IMG_SIZE / max(h, w)
    new_w, new_h = max(int(w*scale), 1), max(int(h*scale), 1)
    img_resized = cv2.resize(img, (new_w, new_h))
    canvas = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    y = (IMG_SIZE - new_h) // 2
    x = (IMG_SIZE - new_w) // 2
    canvas[y:y+new_h, x:x+new_w] = img_resized
    return canvas

def split_lines_dynamically(bin_img, min_height=20):
    proj = np.sum(bin_img, axis=1)
    lines = []
    in_line = False
    start = 0
    threshold = 10 
    
    for y, val in enumerate(proj):
        if val > threshold and not in_line:
            in_line = True
            start = y
        elif val <= threshold and in_line:
            in_line = False
            end = y
            if (end - start) > min_height:
                lines.append(bin_img[start:end, :])
    if in_line:
        lines.append(bin_img[start:, :])
    return lines

def extract_chars_from_two_lines(line1, line2, letter=None, min_area=20):
    combined = np.vstack([line1, line2])
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 1. Get initial bounding boxes
    rects = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w*h < min_area:
            continue
        rects.append([x, y, w, h])

    if not rects:
        return []

    # 2. Merge overlapping components (for i and j with dots)
    rects = sorted(rects, key=lambda r: r[0])
    merged_rects = []
    used = set()

    for i in range(len(rects)):
        if i in used: continue
        curr_x, curr_y, curr_w, curr_h = rects[i]
        
        for j in range(len(rects)):
            if i == j or j in used: continue
            next_x, next_y, next_w, next_h = rects[j]

            # If they overlap horizontally, they are part of the same letter (like the dots of i and j)
            x_overlap = not (next_x > curr_x + curr_w + 2 or curr_x > next_x + next_w + 2)
            
            if x_overlap:
                # Expand current box to cover the new piece
                new_x = min(curr_x, next_x)
                new_y = min(curr_y, next_y)
                new_w = max(curr_x + curr_w, next_x + next_w) - new_x
                new_h = max(curr_y + curr_h, next_y + next_h) - new_y
                
                curr_x, curr_y, curr_w, curr_h = new_x, new_y, new_w, new_h
                used.add(j)

        merged_rects.append([curr_x, curr_y, curr_w, curr_h])
        used.add(i)

    # 3. Final extraction and padding
    merged_rects = sorted(merged_rects, key=lambda r: r[0])
    final_chars = []
    for x, y, w, h in merged_rects:
        char_img = combined[y:y+h, x:x+w]
        padded = resize_and_pad(char_img)
        if padded is not None:
            final_chars.append(padded)
            
    return final_chars

def build_alphabet_dataset(alphabet_img_path, out_dir):
    img = cv2.imread(alphabet_img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {alphabet_img_path}")
    bin_img = preprocess(img)
    
    # Dynamically find lines
    lines = split_lines_dynamically(bin_img)
    
    alphabet = list(string.ascii_lowercase)
    os.makedirs(out_dir, exist_ok=True)
    
    for i, letter in enumerate(alphabet):
        if (2*i + 1) >= len(lines):
            print(f"Finished at letter {letter} (out of lines)")
            break
            
        line1 = lines[2*i]
        line2 = lines[2*i+1]
        
        chars = extract_chars_from_two_lines(line1, line2, letter=letter)
        
        if len(chars) == 0:
            print(f"Warning: no chars found for {letter}")
            continue
            
        letter_dir = os.path.join(out_dir, letter)
        os.makedirs(letter_dir, exist_ok=True)
        for idx, c in enumerate(chars):
            cv2.imwrite(os.path.join(letter_dir, f"{letter}_{idx:03d}.png"), c)
            