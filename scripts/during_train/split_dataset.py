import os
import random
import shutil
from pathlib import Path

# Đường dẫn dataset của bạn
DATASET_DIR = Path("datasets/.main_datasets")

train_images_dir = DATASET_DIR / "train" / "images"
train_labels_dir = DATASET_DIR / "train" / "labels"

valid_images_dir = DATASET_DIR / "valid" / "images"
valid_labels_dir = DATASET_DIR / "valid" / "labels"

test_images_dir = DATASET_DIR / "test" / "images"
test_labels_dir = DATASET_DIR / "test" / "labels"

# Tạo thư mục valid/test nếu chưa có
for folder in [
    valid_images_dir,
    valid_labels_dir,
    test_images_dir,
    test_labels_dir,
]:
    folder.mkdir(parents=True, exist_ok=True)

# Lấy danh sách ảnh trong train/images
image_extensions = [".jpg", ".jpeg", ".png"]
images = [
    img for img in train_images_dir.iterdir()
    if img.suffix.lower() in image_extensions
]

random.seed(42)
random.shuffle(images)

total = len(images)

# Tỷ lệ chia
valid_ratio = 0.2
test_ratio = 0.1

valid_count = int(total * valid_ratio)
test_count = int(total * test_ratio)

valid_images = images[:valid_count]
test_images = images[valid_count:valid_count + test_count]

def move_image_and_label(image_path, target_images_dir, target_labels_dir):
    label_path = train_labels_dir / f"{image_path.stem}.txt"

    # Move image
    shutil.move(str(image_path), str(target_images_dir / image_path.name))

    # Move label nếu có
    if label_path.exists():
        shutil.move(str(label_path), str(target_labels_dir / label_path.name))
    else:
        print(f"Cảnh báo: Không tìm thấy label cho {image_path.name}")

for img in valid_images:
    move_image_and_label(img, valid_images_dir, valid_labels_dir)

for img in test_images:
    move_image_and_label(img, test_images_dir, test_labels_dir)

print("Chia dataset xong.")
print(f"Tổng ảnh ban đầu: {total}")
print(f"Valid: {len(valid_images)}")
print(f"Test: {len(test_images)}")
print(f"Train còn lại: {len(list(train_images_dir.iterdir()))}")