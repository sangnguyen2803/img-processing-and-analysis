"""
train_models.py
Trains two classifiers (SVM and Random Forest) for handwritten character recognition. 
Feature vectors are built using a combination of HOG descriptors and Hu moments.
"""

import os
import cv2
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import dump
from features import extract_features

# Paths anf folders
DATA_PATH = "data/chars"   # folder with subfolders a, b, ..., z
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Load dataset
def load_dataset(path):
    X, y = [], []
    
    # This gets the list of labels from folder names and then sort them alphabetically
    labels = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
    
    for label in labels:
        label_dir = os.path.join(path, label)
        for img_name in os.listdir(label_dir):
            if img_name.startswith('.'):
                continue
            img_path = os.path.join(label_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                continue
                
            feat = extract_features(img, visualize=False)
            
            if feat is not None:
                X.append(feat)
                y.append(label)
    
    X = np.array(X)
    y = np.array(y)

    # Create a DataFrame and export features to CSV
    # Generate feature names:
    # - HOG features first
    # - Followed by the 7 Hu moments
    feature_names = [f"feat_{i}" for i in range(X.shape[1] - 7)] + [f"hu_{i+1}" for i in range(7)]
    
    df = pd.DataFrame(X, columns=feature_names)
    df['label'] = y  # Add class label as the last column
    
    # 1. Data preview
    print(df.head())
    print(f"\nNumber of samples: {len(df)}")
    print(f"Number of features: {X.shape[1]}")
    
    # 2. Save features to csv
    csv_path = "dataset_features.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nFeatures exported to: {csv_path}")
    
    return X, y

# Train and evaluate models (train/test split)
def train_models_evaluation(data_path):
    X, y = load_dataset(data_path)
    
    # Split into train set (80%) and test set (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # SVM Evaluation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    svm_clf = SVC(kernel='rbf', C=10, class_weight='balanced')
    svm_clf.fit(X_train_scaled, y_train)
    
    print("\n[SVM Classification Report]")
    y_pred_svm = svm_clf.predict(X_test_scaled)
    print(classification_report(y_test, y_pred_svm))

    #Random Forest Evaluation 
    rf_clf = RandomForestClassifier(n_estimators=300, random_state=42)
    rf_clf.fit(X_train, y_train)
    
    print("\n[Random Forest Classification Report]")
    y_pred_rf = rf_clf.predict(X_test)
    print(classification_report(y_test, y_pred_rf))

# Train final models on the full dataset
def train_models(data_path):
    X, y = load_dataset(data_path)
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")

    # Feature scaling for SVM only
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train SVM
    svm_clf = SVC(
        kernel='rbf',
        C=10,
        gamma='scale',
        class_weight='balanced',
        probability=False,
        random_state=42
    )
    svm_clf.fit(X_scaled, y)
    dump(svm_clf, os.path.join(MODELS_DIR, "svm_model.joblib"))
    dump(scaler, os.path.join(MODELS_DIR, "svm_scaler.joblib"))

    # Train Random Forest
    rf_clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42
    )
    rf_clf.fit(X, y)  # RF does not need scaling
    dump(rf_clf, os.path.join(MODELS_DIR, "rf_model.joblib"))