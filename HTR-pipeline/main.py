import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from segmentation import build_alphabet_dataset
from augment import process_dataset
from benchmark import run_benchmark
from train import train_models, train_models_evaluation
from recognize import recognize_text, evaluate

# STEP 1 – Build the character dataset
# build_alphabet_dataset(
#     "data/raw/alphabet_1.png",
#     "data/chars"
# )

# EXTRA STEP - augment image (run once only)
# process_dataset("data/chars")

# COMPARISON STEP - compare different feature extraction choices
# run_benchmark()

# STEP 2 – training
# train_models("data/chars")
train_models_evaluation("data/chars")

# STEP 3 – recognition

svm_text, rf_text = recognize_text("data/raw/full_text_1.jpg")

print("SVM:", "".join(svm_text))
print("RF :", "".join(rf_text))
