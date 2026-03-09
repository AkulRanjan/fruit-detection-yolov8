# 🍎 Fruit Detection using YOLOv8

Real-time fruit detection and segmentation system built with **YOLOv8** (Ultralytics). Detects and classifies **5 fruit types** using a custom-trained model.

---

## 🎯 Detected Classes

| # | Fruit |
|---|-------|
| 0 | Apple |
| 1 | Banana |
| 2 | Mango |
| 3 | Orange |
| 4 | Watermelon |

---

## 📁 Project Structure

```
├── balanceall.py          # Balance dataset across classes
├── camera.py              # Real-time inference using webcam/Pi camera
├── check_images.py        # Validate image-label pairs
├── class_result.py        # Analyze per-class detection results
├── fix_val2.py            # Fix validation set issues
├── merger.py              # Merge multiple fruit datasets
├── result.py              # Evaluate model results
├── swap_train.py          # Swap/reorganize training splits
├── swap_val2.py           # Swap/reorganize validation splits
├── verify_final.py        # Final verification of dataset integrity
├── whathappen.py          # Debug / inspection utility
├── fruit_final2.yaml      # YOLOv8 dataset configuration
├── FruitDetection_README.docx  # Detailed project documentation
├── requirements.txt       # Python dependencies
└── .gitignore
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/AkulRanjan/fruit-detection-yolov8.git
cd fruit-detection-yolov8

# Create virtual environment
python -m venv fruitenv
source fruitenv/bin/activate       # Linux/Mac
fruitenv\Scripts\activate          # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 🏋️ Training

```bash
yolo segment train data=fruit_final2.yaml model=yolov8n-seg.pt epochs=50 imgsz=640
```

> **Note:** Download `yolov8n-seg.pt` from [Ultralytics](https://docs.ultralytics.com/models/yolov8/).

---

## 🔍 Inference (Real-time Camera)

```bash
python camera.py
```

Press **`q`** to quit the camera feed.

---

## 📦 Model Weights

Model weights are **not included** in this repository due to file size.

Download the trained weights from:
> 🔗 *[Add your Google Drive / HuggingFace link here]*

Place the downloaded `.pt` file in the project root directory.

---

## 📊 Dataset

The dataset was sourced from [Roboflow](https://roboflow.com/) and includes:

- **Apple** (apple-grade-seg)
- **Banana** (banana_segmentation)
- **Mango**
- **Orange**
- **Watermelon**

Datasets are **not included** in this repository. Download from:
> 🔗 *[Add your Roboflow / Kaggle / Drive link here]*

---

## 🧪 Utility Scripts

| Script | Purpose |
|--------|---------|
| `balanceall.py` | Ensures balanced class distribution in the dataset |
| `check_images.py` | Validates that images have matching label files |
| `merger.py` | Merges individual fruit datasets into one unified dataset |
| `fix_val2.py` | Fixes issues in the validation split |
| `swap_train.py` / `swap_val2.py` | Reorganizes train/val splits |
| `verify_final.py` | Final integrity check before training |
| `class_result.py` / `result.py` | Analyzes detection results and metrics |
| `whathappen.py` | Debugging and inspection tool |

---

## 🛠️ Tech Stack

- **Model**: YOLOv8n-seg (Ultralytics)
- **Language**: Python 3.8+
- **Libraries**: OpenCV, NumPy, Ultralytics
- **Hardware**: Trained on GPU, deployable on Raspberry Pi 5

---

## 📝 License

This project is for educational purposes.

---

## 👤 Author

**Akul Ranjan**
- GitHub: [@AkulRanjan](https://github.com/AkulRanjan)
