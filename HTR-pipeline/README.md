## Table of Contents

- [Summary](#summary)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Benchmarking](#benchmarking)

## Summary

This repository contains code and data for training and evaluating character recognition models (SVM, Random Forest) on a dataset of lowercase characters. 
The `src` directory holds preprocessing, feature extraction, training, recognition, segmentation, augmentation, and benchmarking scripts.

## Project Structure

- `data/` - raw data and character images used for training
- `dataset/` - processed datasets
- `segmented_chars/` - output of segmentation routines
- `models/` - serialized model files (e.g. `rf_model.joblib`, `svm_model.joblib`)
- `src/` - main project code:
  - `augment.py` - augmentation helpers
  - `features.py` - feature extraction
  - `segmentation.py` - character segmentation
  - `train.py` - training script
  - `recognize.py` - recognition/inference script
  - `benchmark.py` - evaluation and benchmarking utilities
- `main.py` - top-level orchestration (quick entry point)
- `dataset_features.csv`, `benchmark_results.csv` - example outputs

## Requirements

- Python 3.8+
- Typical Python packages used by this project include:
  - `numpy`, `scikit-learn`, `joblib`, `opencv-python`, `pillow`, `pandas`, `matplotlib`, `scikit-image`

## Dependency installation

1. To install the primary dependencies via pip, run:
```bash
pip3 install opencv-python numpy scikit-image scikit-learn pandas joblib matplotlib
```
2. Install the dependencies (see Dependency installation).
3. Prepare data under `data/` (the repository includes a `data/chars/` structure for each letter).

## Usage
You can use `python main.py` as a quick entry point; check the script header for supported arguments.

The `models/` directory contains example pre-trained models:

- `rf_model.joblib` - Random Forest model
- `svm_model.joblib` - SVM model
- `svm_scaler.joblib` - scaler used for SVM features


## Benchmarking

Benchmark results are written to `benchmark_results.csv` when running `src/benchmark.py`. The script compares model accuracy and feature extraction performance.