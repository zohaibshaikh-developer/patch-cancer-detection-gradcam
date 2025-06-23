import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

# Settings
SOURCE_ROOT = "/scratch/zs00732/cancer_detection_camelyon17/dataset/camelyon17-clean"
DEST_ROOT = "/scratch/zs00732/cancer_detection_camelyon17/dataset/binary_split"
SPLIT_RATIO = 0.8  # 80% train, 20% val
CLASSES = ["normal", "anomaly"]

# Output directories
train_dir = os.path.join(DEST_ROOT, "train")
val_dir = os.path.join(DEST_ROOT, "val")

# Ensure target structure exists
for split in ["train", "val"]:
    for cls in CLASSES:
        os.makedirs(os.path.join(DEST_ROOT, split, cls), exist_ok=True)

def collect_images():
    image_paths = {"normal": [], "anomaly": []}
    for node_idx in range(5):
        node_path = os.path.join(SOURCE_ROOT, f"node{node_idx}", "test")
        for cls in CLASSES:
            cls_path = os.path.join(node_path, cls)
            if os.path.exists(cls_path):
                images = list(Path(cls_path).glob("*"))
                image_paths[cls].extend(images)
    return image_paths

def split_and_copy(image_paths):
    for cls in CLASSES:
        images = image_paths[cls]
        random.shuffle(images)

        split_idx = int(len(images) * SPLIT_RATIO)
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]

        print(f"🔍 Class: {cls} | Total: {len(images)} | Train: {len(train_imgs)} | Val: {len(val_imgs)}")

        for img in tqdm(train_imgs, desc=f"📥 Copying {cls} to train"):
            shutil.copy(img, os.path.join(train_dir, cls, img.name))
        for img in tqdm(val_imgs, desc=f"📥 Copying {cls} to val"):
            shutil.copy(img, os.path.join(val_dir, cls, img.name))

if __name__ == "__main__":
    print("📂 Collecting images from all nodes...")
    all_images = collect_images()
    print("✂️ Splitting and copying to binary_split/train and binary_split/val")
    split_and_copy(all_images)
    print("✅ Dataset preparation complete.")
