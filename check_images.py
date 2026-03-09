from pathlib import Path
from collections import Counter

label_dir = Path("dataset_final2/labels/train")
class_file_counts = Counter()

for f in label_dir.glob("*.txt"):
    lines = [l.strip() for l in open(f) if l.strip()]
    if not lines:
        continue
    classes = [int(l.split()[0]) for l in lines]
    dominant = Counter(classes).most_common(1)[0][0]
    class_file_counts[dominant] += 1

names = {0: "Apple", 1: "Banana", 2: "Mango", 3: "Orange", 4: "Watermelon"}
print("=== Image Count Per Class ===")
total = 0
for cls in sorted(class_file_counts):
    count = class_file_counts[cls]
    total += count
    print(f"  {names[cls]:12s} → {count:5d} images")
print(f"\n  {'TOTAL':12s} → {total:5d} images")