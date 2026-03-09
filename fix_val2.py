import shutil
import random
from pathlib import Path
from collections import Counter

random.seed(42)

label_dir = Path("dataset_final2/labels/val")
image_dir = Path("dataset_final2/images/val")

class_files = {0: [], 1: [], 2: [], 3: [], 4: []}

for f in label_dir.glob("*.txt"):
    lines = [l.strip() for l in open(f) if l.strip()]
    if not lines:
        continue
    classes = [int(l.split()[0]) for l in lines]
    dominant = Counter(classes).most_common(1)[0][0]
    class_files[dominant].append(f)

names = {0: "Apple", 1: "Banana", 2: "Mango", 3: "Orange", 4: "Watermelon"}

print("=== Val image count per class BEFORE fix ===")
for cls, files in class_files.items():
    print(f"  {names[cls]:12s} → {len(files)} images")

TARGET = min(len(f) for f in class_files.values() if len(f) > 0)
print(f"\n  Target per class → {TARGET} images")

new_label = Path("dataset_final2/labels/val_fixed")
new_image = Path("dataset_final2/images/val_fixed")
new_label.mkdir(parents=True, exist_ok=True)
new_image.mkdir(parents=True, exist_ok=True)

for cls, files in class_files.items():
    selected = random.sample(files, min(TARGET, len(files)))
    copied = 0
    for f in selected:
        for ext in [".jpg", ".png", ".jpeg"]:
            img = image_dir / (f.stem + ext)
            if img.exists():
                shutil.copy(f,   new_label / f.name)
                shutil.copy(img, new_image / img.name)
                copied += 1
                break
    print(f"  ✅ {names[cls]:12s} → {copied} images")

print("\n🎉 Fixed val saved to val_fixed/")