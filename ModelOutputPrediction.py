import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image

# --- 1. DEFINE PATHS & CONSTANTS ---
MODEL_PATH = 'Give Path Of The Model'
VALIDATION_PATH = 'Path Of the images on which you want to predict'

IMG_SHAPE = (150, 150)
NUM_IMAGES_TO_TEST = 500
THRESHOLD = 0.5  

# --- 2. LOAD THE SAVED MODEL ---
print(f"Loading model from {MODEL_PATH}...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"--- ERROR ---")
    print(f"Could not load model. Make sure the file exists at: {MODEL_PATH}")
    print(f"Also ensure you have added the 'SteganoModel' model to this notebook.")
    print(f"Error details: {e}")
    # Stop the script if the model didn't load
    raise SystemExit("Stopping script.")

# --- 3. LOAD AND PREDICT ON IMAGES ---
print(f"\n--- Running predictions on images from: {VALIDATION_STEGO_PATH} ---")
print(f"Using prediction threshold: {THRESHOLD}\n")

# --- NEW: Counter for accuracy ---
correct_predictions = 0

try:
    all_stego_images = os.listdir(VALIDATION_STEGO_PATH)
    image_files = [f for f in all_stego_images if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"--- ERROR ---")
        print(f"No image files found in directory: {VALIDATION_STEGO_PATH}")
        print("Please check the path.")
        raise SystemExit("Stopping script.")

    # Loop over the first few images
    for i, img_name in enumerate(image_files[:NUM_IMAGES_TO_TEST]):
        img_path = os.path.join(VALIDATION_STEGO_PATH, img_name)
        
        # 1. Load the image
        img = image.load_img(
            img_path, 
            target_size=IMG_SHAPE, 
            color_mode='rgb'
        )
        
        # 2. Preprocess the image
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0  # Rescale
        
        # 3. Create a batch
        img_batch = np.expand_dims(img_array, axis=0)
        
        # 4. Predict
        prediction = model.predict(img_batch, verbose=0) # verbose=0 hides the 1/1 progress bar
        
        # 5. Print the result
        prediction_score = prediction[0][0]
        
        # --- NEW: Apply threshold to get the label ---
        if prediction_score > THRESHOLD:
            label = "Stego"
        else:
            label = "Not Stego (Cover)"
            
        print(f"Image [{i+1}/{NUM_IMAGES_TO_TEST}]: {img_name}")
        print(f"    -> Prediction: {prediction_score:.4f}  =>  **{label}**")

except FileNotFoundError:
    print(f"--- ERROR ---")
    print(f"Directory not found: {VALIDATION_STEGO_PATH}")
    print("Please make sure your dataset is added and the path is correct.")
