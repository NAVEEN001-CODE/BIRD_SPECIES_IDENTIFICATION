import os

# Set the path to your training data directory
TRAIN_DIR = r"F:\BIRD SPECIES IDENTIFICATION\dataset\train"

# Get a list of all subdirectories, which are your class names
class_names = [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]

# The ImageDataGenerator sorts the classes alphabetically by default, so we sort them here too.
class_names.sort()

# Print the list in a format you can copy and paste
print("Your class names in the correct order:")
print(class_names)                                       