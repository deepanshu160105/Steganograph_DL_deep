!pip install stegano

import os
import sys
import string
import random
from PIL import Image
from stegano import lsb
from google.colab import drive


# --- 1. PATHS ---
# ⚠️ Update these paths to match your folders in Google Drive!
INPUT_FOLDER_PATH = "Your Clean Images Folder"
OUTPUT_FOLDER_PATH = "Your Stego Images Folder"

# --- 2. CONFIGURATION ---
MIN_PAYLOAD_SIZE = 1126  
MAX_PAYLOAD_SIZE = 4000  # A reasonable upper limit (max for 150x150 is ~8400)
TARGET_SIZE = (150, 150)

# --- 3. SCRIPT FUNCTIONS ---

def generate_random_text(size):
    """Generates a random string of printable ASCII characters."""
    chars = string.ascii_letters + string.digits + string.punctuation + " "
    return ''.join(random.choice(chars) for _ in range(size))

def process_and_embed_images(input_dir, output_dir):
    """
    Reads images from input_dir, filters for 150x150,
    embeds steganography, and saves to output_dir.
    """
    print(f"\n--- STARTING PROCESS ---")
    print(f"Reading from: {input_dir}")
    print(f"Writing to:   {output_dir}")

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    processed_count = 0
    skipped_count = 0

    if not os.path.isdir(input_dir):
        print(f"Error: Input path '{input_dir}' is not a valid directory.")
        return

    for filename in os.listdir(input_dir):
        input_file_path = os.path.join(input_dir, filename)

        # Get the filename parts
        base_name, ext = os.path.splitext(filename)

        # Define the new PNG output path
        output_png_path = os.path.join(output_dir, f"{base_name}_stego.png")

        try:
            img = Image.open(input_file_path)

            # --- Filtering Step ---
            if img.size != TARGET_SIZE:
                print(f"  Skipping {filename}: Size is {img.size}, not {TARGET_SIZE}.")
                skipped_count += 1
                img.close()
                continue

            # --- Embedding Step ---
            # Generate a new random payload for *this* image
            current_payload_size = random.randint(MIN_PAYLOAD_SIZE, MAX_PAYLOAD_SIZE)
            payload = generate_random_text(current_payload_size)

            # The stegano library needs the image in RGB mode
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Embed the secret message
            secret_image = lsb.hide(img, payload)

            # Save the new stego image to the output folder
            secret_image.save(output_png_path)
            img.close()

            print(f"  Processed {filename}: Embedded {current_payload_size} bytes, saved to output folder.")
            processed_count += 1

        except Exception as e:
            # This catches non-image files and other errors
            print(f"  Skipping {filename}: Not a valid image or error occurred. ({e})")
            skipped_count += 1

    print("\n--- ALL OPERATIONS COMPLETE ---")
    print(f"Successfully processed and saved: {processed_count} images.")
    print(f"Skipped (wrong size or not images): {skipped_count} files.")

# --- 5. RUN THE SCRIPT ---
if __name__ == "__main__":
    process_and_embed_images(INPUT_FOLDER_PATH, OUTPUT_FOLDER_PATH)
