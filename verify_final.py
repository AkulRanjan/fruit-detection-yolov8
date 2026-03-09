from pathlib import Path
from collections import Counter

label_dir = Path("dataset_final2/labels/train")
counts = Counter()

for f in label_dir.glob("*.txt"):
    for line in open(f):
        if line.strip():
            counts[int(line.split()[0])] += 1

names = {0: "Apple", 1: "Banana", 2: "Mango", 3: "Orange", 4: "Watermelon"}
print("=== Final Train Image Distribution ===")

train_labels = list(Path("dataset_final2/labels/train").glob("*.txt"))
val_labels   = list(Path("dataset_final2/labels/val").glob("*.txt"))
train_images = list(Path("dataset_final2/images/train").glob("*.*"))
val_images   = list(Path("dataset_final2/images/val").glob("*.*"))

print(f"\n  Train images : {len(train_images)}")
print(f"  Train labels : {len(train_labels)}")
print(f"  Val   images : {len(val_images)}")
print(f"  Val   labels : {len(val_labels)}")

print("\n=== Sanity Check ===")
if len(train_labels) == len(train_images):
    print("  ✅ Train labels and images match")
else:
    print(f"  ❌ MISMATCH — labels: {len(train_labels)} | images: {len(train_images)}")

if len(val_labels) == len(val_images):
    print("  ✅ Val labels and images match")
else:
    print(f"  ❌ MISMATCH — labels: {len(val_labels)} | images: {len(val_images)}")