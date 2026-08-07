import shutil
import random
from pathlib import Path
from collections import Counter

random.seed(42)

label_dir = Path("dataset_final2/labels/train")
image_dir = Path("dataset_final2/images/train")

# ─── Group files by dominant class ────────────────────────────────
class_files = {0: [], 1: [], 2: [], 3: [], 4: []}

for f in label_dir.glob("*.txt"):
    lines = [l.strip() for l in open(f) if l.strip()]
    if not lines:
        continue
    classes = [int(l.split()[0]) for l in lines]
    dominant = Counter(classes).most_common(1)[0][0]
    class_files[dominant].append(f)

names = {0: "Apple", 1: "Banana", 2: "Mango", 3: "Orange", 4: "Watermelon"}

print("=== Image files per class before balancing ===")
for cls, files in class_files.items():
    print(f"  {names[cls]:12s} → {len(files)} images")

# ─── Target = 1500 images per class ───────────────────────────────
TARGET = 1500

output_label = Path("dataset_final2/labels/train_balanced")
output_image = Path("dataset_final2/images/train_balanced")
output_label.mkdir(parents=True, exist_ok=True)
output_image.mkdir(parents=True, exist_ok=True)

for cls, files in class_files.items():
    if len(files) == 0:
        print(f"  ⚠️  {names[cls]} has 0 files — skipping")
        continue

    if len(files) >= TARGET:
        selected = random.sample(files, TARGET)
    else:
        # Oversample by repeating
        selected = files * (TARGET // len(files)) + \
                   random.sample(files, TARGET % len(files))

    copied = 0
    for i, f in enumerate(selected):
        img_path = None
        for ext in [".jpg", ".png", ".jpeg"]:
            candidate = image_dir / (f.stem + ext)
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            continue

        new_stem = f"{f.stem}_{i}"
        shutil.copy(f,        output_label / (new_stem + ".txt"))
        shutil.copy(img_path, output_image / (new_stem + ext))
        copied += 1

    print(f"  ✅ {names[cls]:12s} → {copied} images")

print("\n🎉 Balanced train saved to train_balanced/")