import os
import shutil
import random
from google.colab import drive

# --- 1. CONFIGURE YOUR PATHS ---
# ⚠️ Update these paths!

# Input folders (where your original images are)
CLEAN_FOLDER_PATH = "Your Clean folder path"
STEGO_FOLDER_PATH = "Your Stego folder path"

# Output folder (where your new dataset will be created)
BASE_DATASET_PATH = "Output Folder Path"

# How much data to set aside for validation/testing
VALIDATION_SPLIT_PERCENT = 0.2  # (20%)
# --------------------------------

def split_files(source_folder, train_dest, val_dest, split_ratio):
    """
    Randomly splits files from a source folder into train and validation
    destination folders and copies them.
    """
    if not os.path.exists(source_folder):
        print(f"Error: Source folder does not exist: {source_folder}")
        return 0, 0

    # Get all file names
    files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    random.shuffle(files)

    # Calculate the split index
    split_index = int(len(files) * split_ratio)

    # Get file lists for validation and training
    validation_files = files[:split_index]
    training_files = files[split_index:]

    # Copy validation files
    for f in validation_files:
        shutil.copy(os.path.join(source_folder, f), val_dest)

    # Copy training files
    for f in training_files:
        shutil.copy(os.path.join(source_folder, f), train_dest)

    return len(training_files), len(validation_files)

def create_dataset_structure():
    print("--- Starting Dataset Creation ---")

    # 1. Define all the new paths
    train_dir = os.path.join(BASE_DATASET_PATH, 'train')
    val_dir = os.path.join(BASE_DATASET_PATH, 'validation')

    train_clean_dir = os.path.join(train_dir, '0_clean')
    train_stego_dir = os.path.join(train_dir, '1_stego')

    val_clean_dir = os.path.join(val_dir, '0_clean')
    val_stego_dir = os.path.join(val_dir, '1_stego')

    # 2. Create all directories
    for path in [train_clean_dir, train_stego_dir, val_clean_dir, val_stego_dir]:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")
        else:
            print(f"Directory already exists: {path}")

    # 3. Split and copy clean images
    print(f"\nSplitting 'clean' images...")
    clean_train, clean_val = split_files(CLEAN_FOLDER_PATH,
                                         train_clean_dir,
                                         val_clean_dir,
                                         VALIDATION_SPLIT_PERCENT)
    print(f"Clean: {clean_train} train, {clean_val} validation.")

    # 4. Split and copy stego images
    print(f"\nSplitting 'stego' images...")
    stego_train, stego_val = split_files(STEGO_FOLDER_PATH,
                                         train_stego_dir,
                                         val_stego_dir,
                                         VALIDATION_SPLIT_PERCENT)
    print(f"Stego: {stego_train} train, {stego_val} validation.")

    print("\n--- Dataset folder structure is complete! ---")
    print(f"Total Training Images:   {clean_train + stego_train}")
    print(f"Total Validation Images: {clean_val + stego_val}")

# --- 5. RUN THE SCRIPT ---
if __name__ == "__main__":
    create_dataset_structure()
