import os
from scripts.preprocess_data import preprocess_images
from scripts.train_model import train_model

RAW_DATA_DIR = 'data'  # Input directory
OUTPUT_DATA_DIR = 'data_processed'  # Output directory
MODEL_PATH = 'models/cat_classifier.pth'

# Ensure the model exists or train it
def ensure_model_exists():
    if not os.path.exists(MODEL_PATH):
        print("Model not found! Training a new model...")
        preprocess_images(RAW_DATA_DIR, OUTPUT_DATA_DIR)
        train_model(OUTPUT_DATA_DIR, MODEL_PATH)
    else:
        print("Model found! Skipping training.")

# Main entry point
def main():
    print("Welcome to 'Is Dat My Kitty?'")
    ensure_model_exists()

if __name__ == "__main__":
    main()
