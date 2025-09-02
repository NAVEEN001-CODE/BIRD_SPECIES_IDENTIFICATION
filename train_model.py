import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ===============================
# PATHS
# ===============================
TRAIN_DIR = r"F:\BIRD SPECIES IDENTIFICATION\dataset\train"
TEST_DIR  = r"F:\BIRD SPECIES IDENTIFICATION\dataset\test"
MODEL_PATH = "bird_species_model.h5"

# ===============================
# DATASET PREPARATION
# ===============================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.3,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# ===============================
# MODEL CREATION (MobileNetV2)
# ===============================
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Phase 1: freeze base layers

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
predictions = Dense(train_generator.num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=0.001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# ===============================
# CALLBACKS
# ===============================
callbacks = [
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    ModelCheckpoint("best_model.h5", monitor="val_accuracy", save_best_only=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=1e-7)
]

# ===============================
# TRAINING (Phase 1: Head only)
# ===============================
print("🔄 Phase 1: Training classifier head (10 epochs, frozen base)")
history1 = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=10,
    callbacks=callbacks,
    verbose=1
)

# ===============================
# FINE-TUNING (Phase 2: Unfreeze base)
# ===============================
print("🔄 Phase 2: Fine-tuning MobileNetV2 (50 epochs, unfrozen last layers)")

for layer in base_model.layers[-60:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-5),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

history2 = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=50,
    callbacks=callbacks,
    verbose=1
)

# ===============================
# SAVE MODEL
# ===============================
model.save(MODEL_PATH)
print(f"✅ Final model saved as {MODEL_PATH}")
                                                                                                                                                                                                                                                                                                                                    