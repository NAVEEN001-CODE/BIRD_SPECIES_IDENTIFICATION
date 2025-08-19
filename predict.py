import sys
import os
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model

# Path to your model
model_path = r"F:\BIRD SPECIES IDENTIFICATION\bird_species_model.h5"

# Path to training folder (to get class names)
train_dir = r"F:\BIRD SPECIES IDENTIFICATION\dataset\train"

# Check if model file exists
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    sys.exit(1)

# Load the trained model
model = load_model(model_path)

# Load class names automatically from dataset folder
if not os.path.exists(train_dir):
    print(f"Error: Training directory not found at {train_dir}")
    sys.exit(1)

class_names = sorted(os.listdir(train_dir))
print(f"✅ Classes loaded: {class_names}")

# Get image path from command-line argument
if len(sys.argv) != 2:
    print("Usage: python predict.py <image_path>")
    sys.exit(1)

img_path = sys.argv[1]

# Check if image file exists
if not os.path.exists(img_path):
    print(f"Error: Image file not found at {img_path}")
    sys.exit(1)

# Load and preprocess the image
img = image.load_img(img_path, target_size=(224, 224))  # adjust to your model input size
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Make prediction
pred = model.predict(img_array)
pred_class = np.argmax(pred, axis=1)[0]
confidence = pred[0][pred_class] * 100

# Print result
print(f"Predicted Bird: {class_names[pred_class]}, Confidence: {confidence:.2f}%")
