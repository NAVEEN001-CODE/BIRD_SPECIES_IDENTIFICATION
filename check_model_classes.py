import os
from tensorflow.keras.models import load_model

# Load your saved model
model = load_model("bird_species_model.h5")
print("Number of output classes in model:", model.output_shape[-1])

# Path to your dataset's training folder
train_dir = os.path.join("dataset", "train")
current_classes = sorted(os.listdir(train_dir))
print("Number of current classes in dataset:", len(current_classes))

# Optional: print the class names
print("Current dataset classes:", current_classes)
