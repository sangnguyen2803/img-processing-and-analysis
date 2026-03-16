import os
import cv2
import time
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from features import extract_hu, extract_hog, extract_geometric

DATA_PATH = "data/chars"

def get_feature_extractor(method):
    if method == "Geometric":
        return lambda img: extract_geometric(img)
    elif method == "Hu":
        return lambda img: extract_hu(img)
    elif method == "HOG":
        return lambda img: extract_hog(img)
    elif method == "Combined":
        return lambda img: np.hstack([extract_hog(img), extract_hu(img)])

def run_benchmark():
    methods = ["Geometric", "Hu", "HOG", "Combined"]
    results = []

    images, labels = [], []
    for label in sorted(os.listdir(DATA_PATH)):
        ldir = os.path.join(DATA_PATH, label)
        if not os.path.isdir(ldir): continue
        for iname in os.listdir(ldir):
            img = cv2.imread(os.path.join(ldir, iname), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                labels.append(label)

    # Run benchmark for each feature extraction method
    for m in methods:
        print(f"\nEvaluating method: {m}")
        extractor = get_feature_extractor(m)
        
        # 1. Speed
        start_time = time.time()
        features = [extractor(img) for img in images]
        end_time = time.time()
        
        avg_time = (end_time - start_time) / len(images) * 1000 # ms/image
        X = np.array(features)
        y = np.array(labels)

        # Standardization for SVM
        X_scaled = StandardScaler().fit_transform(X)

        # 2. Use cross-validation for evaluation
        svm = SVC(kernel='rbf', C=10)
        rf = RandomForestClassifier(n_estimators=100)
        
        svm_acc = cross_val_score(svm, X_scaled, y, cv=5).mean()
        rf_acc = cross_val_score(rf, X, y, cv=5).mean()

        results.append({
            "Method": m,
            "Feature_Count": X.shape[1],
            "Avg_Time_ms": round(avg_time, 4),
            "SVM_Accuracy": round(svm_acc, 4),
            "RF_Accuracy": round(rf_acc, 4)
        })

    df_res = pd.DataFrame(results)
    print("BENCHMARK RESULTS (PERFORMANCE & EFFICIENCY)")
    print(df_res.to_string(index=False))
    df_res.to_csv("benchmark_results.csv", index=False)