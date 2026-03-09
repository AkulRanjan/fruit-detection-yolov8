import shutil
from pathlib import Path

shutil.copytree("dataset_final2/images/val", "dataset_final2/images/val_old")
shutil.copytree("dataset_final2/labels/val", "dataset_final2/labels/val_old")
print("✅ Original val backed up")

shutil.rmtree("dataset_final2/images/val")
shutil.rmtree("dataset_final2/labels/val")

shutil.move("dataset_final2/images/val_fixed", "dataset_final2/images/val")
shutil.move("dataset_final2/labels/val_fixed", "dataset_final2/labels/val")
print("✅ val_fixed is now val")