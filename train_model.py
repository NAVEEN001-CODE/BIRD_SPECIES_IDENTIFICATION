import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Paths
train_dir = os.path.join("dataset", "train")
test_dir = os.path.join("dataset", "test")

# Data generators (augmentation for training, only rescale for testing)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir, target_size=(224,224), batch_size=32, class_mode='categorical'
)
test_data = test_datagen.flow_from_directory(
    test_dir, target_size=(224,224), batch_size=32, class_mode='categorical'
)

# Base model
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))
base_model.trainable = False   # freeze base layers initially

# Add custom head
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(train_data.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train first stage
history = model.fit(train_data, validation_data=test_data, epochs=15)

# Unfreeze some base layers (fine-tuning)
base_model.trainable = True
for layer in base_model.layers[:100]:  # keep first 100 layers frozen
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),  # lower LR for fine-tuning
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train fine-tuning stage
history_fine = model.fit(train_data, validation_data=test_data, epochs=20)

# Save final model
model.save("bird_species_model.h5")
print("✅ Model saved as bird_species_model.h5")
