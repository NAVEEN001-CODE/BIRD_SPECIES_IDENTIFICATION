import os
import shutil

# Path to your dataset
DATASET_DIR = r"F:\BIRD SPECIES IDENTIFICATION\CUB_200_2011"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
SPLIT_FILE = os.path.join(DATASET_DIR, "train_test_split.txt")
IMAGE_LIST_FILE = os.path.join(DATASET_DIR, "images.txt")

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")

# Create train/ and test/ folders
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# Load image IDs to filenames
id_to_file = {}
with open(IMAGE_LIST_FILE, "r") as f:
    for line in f:
        image_id, filename = line.strip().split()
        id_to_file[int(image_id)] = filename

# Load split info
split_info = {}
with open(SPLIT_FILE, "r") as f:
    for line in f:
        image_id, is_train = line.strip().split()
        split_info[int(image_id)] = int(is_train)

# Copy files into train/ and test/
for img_id, filename in id_to_file.items():
    src_path = os.path.join(IMAGES_DIR, filename)
    class_name = filename.split("/")[0]  # folder name = bird species

    if split_info[img_id] == 1:  # training
        dst_dir = os.path.join(TRAIN_DIR, class_name)
    else:  # testing
        dst_dir = os.path.join(TEST_DIR, class_name)

    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy(src_path, dst_dir)

print("✅ Dataset prepared! Train and Test folders are ready.")
