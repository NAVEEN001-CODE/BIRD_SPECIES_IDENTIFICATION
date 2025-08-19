import os
import shutil

# Starting point — change this to where you extracted the dataset
ROOT_DIR = r"F:\BIRD SPECIES IDENTIFICATION\CUB_200_2011"

# Step 1: Auto-find the folder containing images.txt
def find_base_dir(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "images.txt" in filenames and "image_class_labels.txt" in filenames and "train_test_split.txt" in filenames:
            return dirpath
    return None

BASE_DIR = find_base_dir(ROOT_DIR)
if BASE_DIR is None:
    raise FileNotFoundError("❌ Could not find images.txt in the dataset folder. Check your extraction.")

IMAGES_DIR = os.path.join(BASE_DIR, "images")
OUTPUT_DIR = os.path.join(ROOT_DIR, "dataset")

# Step 2: Create output directories
train_dir = os.path.join(OUTPUT_DIR, "train")
test_dir = os.path.join(OUTPUT_DIR, "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Step 3: Read helper files
images_file = os.path.join(BASE_DIR, "images.txt")
labels_file = os.path.join(BASE_DIR, "image_class_labels.txt")
split_file = os.path.join(BASE_DIR, "train_test_split.txt")

# Mapping: image_id → file path
image_id_to_file = {}
with open(images_file, "r") as f:
    for line in f:
        img_id, img_path = line.strip().split()
        image_id_to_file[int(img_id)] = img_path

# Mapping: image_id → label number
image_id_to_label = {}
with open(labels_file, "r") as f:
    for line in f:
        img_id, label = line.strip().split()
        image_id_to_label[int(img_id)] = int(label)

# Mapping: image_id → train/test split
image_id_to_split = {}
with open(split_file, "r") as f:
    for line in f:
        img_id, is_train = line.strip().split()
        image_id_to_split[int(img_id)] = bool(int(is_train))

# Step 4: Create class folder mapping
label_to_name = {}
for file_name in os.listdir(IMAGES_DIR):
    label_num = int(file_name.split('.')[0])
    label_to_name[label_num] = file_name

for label_num, folder_name in label_to_name.items():
    os.makedirs(os.path.join(train_dir, folder_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, folder_name), exist_ok=True)

# Step 5: Copy images into train/test folders
count_train = 0
count_test = 0

for img_id, file_path in image_id_to_file.items():
    label_num = image_id_to_label[img_id]
    folder_name = label_to_name[label_num]

    src_path = os.path.join(IMAGES_DIR, file_path)
    if not os.path.exists(src_path):
        continue

    if image_id_to_split[img_id]:
        dst_path = os.path.join(train_dir, folder_name, os.path.basename(file_path))
        count_train += 1
    else:
        dst_path = os.path.join(test_dir, folder_name, os.path.basename(file_path))
        count_test += 1

    shutil.copy2(src_path, dst_path)

print(f"✅ Dataset prepared successfully!")
print(f"Training images: {count_train}")
print(f"Testing images: {count_test}")
print(f"Saved to: {OUTPUT_DIR}")
