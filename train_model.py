import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# -------------------------
# CONFIG
# -------------------------
IMG_SIZE = 160
BATCH_SIZE = 32
EPOCHS_PHASE1 = 25
EPOCHS_PHASE2 = 15

TRAIN_DIR = "data/rafdb/train"
VAL_DIR = "data/rafdb/test"

MODEL_PATH = "emotion_mobilenet_rafdb.h5"

# -------------------------
# DATA GENERATORS
# -------------------------
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True,
    zoom_range=0.1,
    brightness_range=[0.8, 1.2]
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# -------------------------
# CLASS WEIGHTS (FIXED)
# -------------------------
labels = train_generator.classes

class_weights_arr = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)

class_weights = dict(
    zip(np.unique(labels), class_weights_arr)
)

print("Class weights:", class_weights)

# -------------------------
# MODEL LOAD OR CREATE
# -------------------------
if os.path.exists(MODEL_PATH):
    print("\n📦 Loading existing model...")
    model = load_model(MODEL_PATH)
else:
    print("\n🆕 Creating new model...")

    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.6)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)

    output = Dense(7, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)

# -------------------------
# COMPILE (PHASE 1 OR RESUME)
# -------------------------
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

model.summary()

# -------------------------
# CALLBACKS
# -------------------------
callbacks = [
    EarlyStopping(patience=4, restore_best_weights=True),
    ReduceLROnPlateau(patience=2, factor=0.3, verbose=1)
]

# -------------------------
# TRAIN PHASE 1 / RESUME
# -------------------------
print("\n=== TRAINING / PHASE 1 ===")

model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_PHASE1,
    class_weight=class_weights,
    callbacks=callbacks
)

# -------------------------
# FINE-TUNING
# -------------------------
print("\n=== PHASE 2: Fine-tuning ===")

for layer in model.layers[:-40]:
    layer.trainable = False

for layer in model.layers[-40:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_PHASE2,
    class_weight=class_weights,
    callbacks=callbacks
)

# -------------------------
# SAVE MODEL
# -------------------------
model.save(MODEL_PATH)

print("\n✅ Training complete. Model saved.")