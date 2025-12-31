import os
import random
import shutil

TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
VAL_RATIO = 0.2   # 20% of images go to validation

os.makedirs(VAL_DIR, exist_ok=True)

for class_name in os.listdir(TRAIN_DIR):
    class_train_dir = os.path.join(TRAIN_DIR, class_name)
    if not os.path.isdir(class_train_dir):
        continue

    images = os.listdir(class_train_dir)
    random.shuffle(images)

    val_count = int(len(images) * VAL_RATIO)
    val_images = images[:val_count]

    class_val_dir = os.path.join(VAL_DIR, class_name)
    os.makedirs(class_val_dir, exist_ok=True)

    for img in val_images:
        src = os.path.join(class_train_dir, img)
        dst = os.path.join(class_val_dir, img)
        shutil.move(src, dst)

print("✅ Validation split created successfully")
