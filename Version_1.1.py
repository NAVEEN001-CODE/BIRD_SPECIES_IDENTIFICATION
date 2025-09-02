# -*- coding: utf-8 -*-
"""
Bird Species Identification using VGG16 (Transfer Learning)
Dataset: CUB-200-2011 (with train/test split)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -------------------------------
# Paths (update if needed)
# -------------------------------
train_path = "F:/BIRD SPECIES IDENTIFICATION/CUB_200_2011/train"
test_path  = "F:/BIRD SPECIES IDENTIFICATION/CUB_200_2011/test"

# -------------------------------
# Image size and base model
# -------------------------------
IMAGE_SIZE = [224, 224]

vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# Freeze pre-trained layers
for layer in vgg.layers:
    layer.trainable = False

# Number of classes = number of folders in train directory
folders = glob(os.path.join(train_path, "*"))

# Add custom layers
x = Flatten()(vgg.output)
prediction = Dense(len(folders), activation='softmax')(x)

# Build model
model = Model(inputs=vgg.input, outputs=prediction)
model.summary()

# Compile model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# -------------------------------
# Data Generators
# -------------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_set = test_datagen.flow_from_directory(
    test_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# -------------------------------
# Train model
# -------------------------------
r = model.fit(
    training_set,
    validation_data=test_set,
    epochs=5,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set)
)

# -------------------------------
# Plot results
# -------------------------------
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss.png')

plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc.png')

# -------------------------------
# Save model
# -------------------------------
model.save('bird_species_vgg16.h5')
print("✅ Model saved as bird_species_vgg16.h5")
