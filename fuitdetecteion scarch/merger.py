import os
import shutil
import random
from pathlib import Path

datasets = [
    {
        "name": "apple",
        "base": "apple-grade-seg.v1i.yolov8",
        "class_map": {0: 0, 1: 0, 2: 0, 3: 0}
    },
    {
        "name": "banana",
        "base": "banana_segmentation.v1i.yolov8",
        "class_map": {0: 1}
    },
    {
        "name": "mango",
        "base": "mango.v1i.yolov8",
        "class_map": {0: 2, 1: 2}
    },
    {
        "name": "orange",
        "base": "orange.v1i.yolov8",
        "class_map": {0: 3, 1: 3}
    },
    {
        "name": "watermelon",
        "base": "Watermelon Detector.v1i.yolov8",   # ← exact folder name
        "class_map": {0: 4}
    },
]

output_base = Path("dataset_final2")
for split in ["train", "val"]:
    (output_base / "images" / split).mkdir(parents=True, exist_ok=True)
    (output_base / "labels" / split).mkdir(parents=True, exist_ok=True)

def remap_label_file(src_path, dst_path, class_map):
    new_lines = []
    with open(src_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            original_class = int(parts[0])
            new_class = class_map.get(original_class, -1)
            if new_class == -1:
                continue
            new_lines.append(str(new_class) + " " + " ".join(parts[1:]))
    if new_lines:
        with open(dst_path, "w") as f:
            f.write("\n".join(new_lines))
        return True
    return False

total_counts = {"train": 0, "val": 0}

for ds in datasets:
    base = Path(ds["base"])
    split_mapping = {
        "train": "train",
        "test":  "train",
        "valid": "val",
    }
    for src_split, dst_split in split_mapping.items():
        img_dir   = base / src_split / "images"
        label_dir = base / src_split / "labels"

        if not img_dir.exists():
            print(f"  ⚠️  Missing: {img_dir} — skipping")
            continue

        image_files = list(img_dir.glob("*.jpg")) + \
                      list(img_dir.glob("*.png")) + \
                      list(img_dir.glob("*.jpeg"))

        copied = 0
        for img_path in image_files:
            label_path = label_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                continue

            unique_name = f"{ds['name']}_{src_split}_{img_path.stem}"
            dst_label = output_base / "labels" / dst_split / (unique_name + ".txt")
            dst_image = output_base / "images" / dst_split / (unique_name + img_path.suffix)

            success = remap_label_file(label_path, dst_label, ds["class_map"])
            if success:
                shutil.copy(img_path, dst_image)
                copied += 1

        total_counts[dst_split] += copied
        print(f"  ✅ {ds['name']:12s} | {src_split:5s} → {dst_split:5s} | {copied} files")

print(f"\n📦 Total train: {total_counts['train']} | val: {total_counts['val']}")