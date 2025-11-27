import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Add, ReLU, GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import sys 

def create_srnet_classifier(input_shape=(150, 150, 3)):
    """
    Creates the SRNet model architecture for classification.
    MODIFIED with Batch Normalization and Dropout.
    """

    inputs = Input(shape=input_shape)

    # Layer 1
    x = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal', name='Layer1')(inputs)
    x = BatchNormalization(name='Layer1_bn')(x) # <-- ADDED BN

    # Layer 2
    x = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal', name='Layer2_conv1')(x)
    x = BatchNormalization(name='Layer2_bn1')(x) # <-- ADDED BN
    x = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal', name='Layer2_conv2')(x)
    x = BatchNormalization(name='Layer2_bn2')(x) # <-- ADDED BN

    # Layer 3
    x = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal', name='Layer3_conv1')(x)
    x = BatchNormalization(name='Layer3_bn1')(x) # <-- ADDED BN
    x = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal', name='Layer3_conv2')(x)
    x = BatchNormalization(name='Layer3_bn2')(x) # <-- ADDED BN

    # Layer 4
    x = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal', name='Layer4_conv1')(x)
    x = BatchNormalization(name='Layer4_bn1')(x) # <-- ADDED BN
    x = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal', name='Layer4_conv2')(x)
    x = BatchNormalization(name='Layer4_bn2')(x) # <-- ADDED BN

    # Layer 5
    x = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal', name='Layer5_conv1')(x)
    x = BatchNormalization(name='Layer5_bn1')(x) # <-- ADDED BN
    x = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal', name='Layer5_conv2')(x)
    x = BatchNormalization(name='Layer5_bn2')(x) # <-- ADDED BN

    # Residual Blocks (Layers 6-12)
    for i in range(6, 13):
        res = x
        x = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal', name=f'Layer{i}_conv1')(x)
        x = BatchNormalization(name=f'Layer{i}_bn1')(x) # <-- ADDED BN

        x = Conv2D(64, (3, 3), padding='same', activation=None, kernel_initializer='he_normal', name=f'Layer{i}_conv2')(x)
        

        x = Add(name=f'Layer{i}_add')([x, res])
        x = ReLU(name=f'Layer{i}_relu')(x)
        x = BatchNormalization(name=f'Layer{i}_bn_out')(x) # <-- ADDED BN

    # Final Conv Blocks (Layers 13-17)
    for i in range(13, 18):
        x = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal', name=f'Layer{i}_conv1')(x)
        x = BatchNormalization(name=f'Layer{i}_bn1')(x) # <-- ADDED BN

        x = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal', name=f'Layer{i}_conv2')(x)
        x = BatchNormalization(name=f'Layer{i}_bn2')(x) # <-- ADDED BN

    x = GlobalAveragePooling2D(name='g_avg_pool')(x)
    x = Dropout(0.5, name='dropout')(x) # <-- ADDED DROPOUT
    outputs = Dense(1, activation='sigmoid', name='output')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


BASE_DATASET_PATH = "/kaggle/input/stegomodeldataset1/StegoModelDataset1"

TRAIN_DIR = os.path.join(BASE_DATASET_PATH, 'train')
VALIDATION_DIR = os.path.join(BASE_DATASET_PATH, 'validation')

IMG_SHAPE = (150, 150, 3) # Your requested RGB shape
BATCH_SIZE = 16

# --- 2. CREATE DATA GENERATORS ---
print("Creating Data Generators...")

datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SHAPE[0], IMG_SHAPE[1]),
    color_mode='rgb',   
    class_mode='binary',
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_generator = datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(IMG_SHAPE[0], IMG_SHAPE[1]),
    color_mode='rgb',
    class_mode='binary',
    batch_size=BATCH_SIZE,
    shuffle=False   
)

# --- 3. CREATE AND COMPILE MODEL  ---
print("Creating and compiling SRNet model...")
model = create_srnet_classifier(input_shape=IMG_SHAPE)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # <-- Significantly Lowered LR
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()


callbacks = [
    # Stop if val_accuracy doesn't improve for 10 epochs
    tf.keras.callbacks.EarlyStopping(
        patience=10,
        monitor='val_accuracy',
        restore_best_weights=True
    ),
    # Save the model after EACH epoch, overwriting the previous one
    tf.keras.callbacks.ModelCheckpoint(
        filepath='/kaggle/working/srnet_rgb_best5.keras',
        monitor='val_accuracy',
        save_best_only=False 
    )
]

# --- 4. TRAIN THE MODEL! ---
print("\n--- STARTING MODEL TRAINING ---")

# Calculate steps per epoch (for validation check only)
steps_per_epoch = train_generator.samples // BATCH_SIZE
validation_steps = validation_generator.samples // BATCH_SIZE

# Check for empty generators
if steps_per_epoch == 0 or validation_steps == 0:
    print("\n--- ERROR ---")
    print("Your generators are empty. This means the paths are wrong or the folders are empty.")
    print(f"Check these paths:")
    print(f"Train path: {TRAIN_DIR}")
    print(f"Validation path: {VALIDATION_DIR}")
    print("\nDid you add the dataset using the '+ Add data' button?")
    sys.exit("Training aborted. Fix folder paths.")


history = model.fit(
    train_generator,
    epochs=100,   
    validation_data=validation_generator,
    callbacks=callbacks
)

print("\n--- TRAINING COMPLETE ---")
print("Model saved/overwritten after each epoch to '/kaggle/working/srnet_rgb_best5.keras'")
