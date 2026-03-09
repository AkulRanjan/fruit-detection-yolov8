import shutil
from pathlib import Path

# Backup original
shutil.copytree("dataset_final2/images/train", "dataset_final2/images/train_old")
shutil.copytree("dataset_final2/labels/train", "dataset_final2/labels/train_old")
print("✅ Original train backed up to train_old")

# Remove original
shutil.rmtree("dataset_final2/images/train")
shutil.rmtree("dataset_final2/labels/train")

# Rename balanced to train
shutil.move("dataset_final2/images/train_balanced", "dataset_final2/images/train")
shutil.move("dataset_final2/labels/train_balanced", "dataset_final2/labels/train")
print("✅ train_balanced is now train")