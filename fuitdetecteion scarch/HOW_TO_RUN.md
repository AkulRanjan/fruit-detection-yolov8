# 🍎 Fruit Detection & Segmentation System — How to Run Guide

## Overview

This project uses a **YOLOv8 segmentation model** trained on 5 fruit classes (Apple, Banana, Mango, Orange, Watermelon) to perform **real-time fruit detection and instance segmentation** via a USB camera feed. The system draws bounding boxes, segmentation masks, confidence scores, and a live fruit-count panel on the video stream.

---

## Prerequisites

| Requirement       | Details                                      |
| ----------------- | -------------------------------------------- |
| **OS**            | Windows 10/11                                |
| **Python**        | 3.11.x (tested with 3.11.9)                 |
| **GPU (optional)**| NVIDIA GPU with CUDA 11.8 (for faster inference) |
| **Camera**        | USB webcam (connected and recognized by Windows) |

> **Note:** The model will work on CPU as well, but GPU (CUDA) provides significantly faster inference for real-time detection.

---

## Project Structure

```
fuitdetecteion scarch/
├── camera.py                          ← Main script (run this)
├── fruitenv/                          ← Python virtual environment
├── runs/segment/fruit_seg_v33/
│   └── weights/
│       ├── best.pt                    ← Best trained model weights
│       └── last.pt                    ← Last epoch model weights
├── fruit_final2.yaml                  ← Dataset config (5 classes)
├── yolov8n-seg.pt                     ← Base YOLOv8 nano-seg model
├── dataset_final2/                    ← Training dataset
├── FruitDetection_README.docx         ← Original project doc
└── HOW_TO_RUN.md                      ← This file
```

---

## Step-by-Step Setup & Run Instructions

### Step 1: Open Terminal

Open **PowerShell** or **Command Prompt** and navigate to the project folder:

```powershell
cd "C:\Users\lakshya sharma\Desktop\cvrelated\fuitdetecteion scarch\fuitdetecteion scarch"
```

---

### Step 2: Activate the Virtual Environment

The project has a pre-built virtual environment called `fruitenv`. Activate it:

**PowerShell:**
```powershell
.\fruitenv\Scripts\Activate.ps1
```

**Command Prompt (cmd):**
```cmd
fruitenv\Scripts\activate.bat
```

> After activation, you should see `(fruitenv)` at the beginning of your terminal prompt.

---

### Step 3: Verify Dependencies (Optional)

All required packages should already be installed in the virtual environment. To verify:

```powershell
pip list
```

**Key packages required:**

| Package          | Version     | Purpose                        |
| ---------------- | ----------- | ------------------------------ |
| `ultralytics`    | 8.4.x       | YOLOv8 framework               |
| `torch`          | 2.7.x+cu118 | PyTorch with CUDA 11.8 support |
| `torchvision`    | 0.22.x      | Vision utilities                |
| `opencv-python`  | 4.13.x      | Camera capture & drawing        |
| `numpy`          | 2.3.x       | Array operations                |

If any package is missing, install it:

```powershell
pip install ultralytics opencv-python numpy
```

For GPU support (CUDA 11.8):
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

### Step 4: Connect USB Camera

1. Plug in your **USB webcam** to any USB port
2. Make sure Windows recognizes the camera (check Device Manager → Cameras)
3. Close any other apps that might be using the camera (Zoom, Teams, etc.)

---

### Step 5: Run the Fruit Detection System

```powershell
python camera.py
```

**What happens:**
1. The YOLOv8 segmentation model loads from `runs/segment/fruit_seg_v33/weights/best.pt`
2. The USB camera opens (index `1`)
3. A window titled **"🍎 Fruit Detection System"** appears with the live feed
4. Detected fruits are shown with:
   - **Colored segmentation masks** (semi-transparent overlay)
   - **Bounding boxes** around each fruit
   - **Labels** with fruit name and confidence score
   - **Fruit Count panel** (top-left corner) showing counts per fruit type

---

### Step 6: Quit the Application

Press the **`Q`** key on your keyboard while the detection window is focused to stop the camera and close the application.

---

## Camera Troubleshooting

### "Cannot open camera" Error

If you see `❌ Cannot open camera`, try changing the camera index in `camera.py`:

```python
# Line 28 in camera.py
cap = cv2.VideoCapture(1)   # USB cam
```

| Index | Camera                              |
| ----- | ----------------------------------- |
| `0`   | Built-in laptop webcam              |
| `1`   | First USB camera (most common)      |
| `2`   | Second USB camera (if multiple)     |

Change the number and re-run if needed.

### Camera is laggy / low FPS

- Ensure you have a **CUDA-capable GPU** and PyTorch with CUDA is installed
- Lower the resolution in `camera.py` (lines 29-30):
  ```python
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
  ```
- Increase confidence threshold to reduce detections:
  ```python
  results = model(frame, conf=0.6, verbose=False)  # was 0.4
  ```

---

## Detected Fruit Classes

| Class ID | Fruit      | Mask Color |
| -------- | ---------- | ---------- |
| 0        | Apple      | 🟢 Green   |
| 1        | Banana     | 🟡 Yellow  |
| 2        | Mango      | 🟠 Orange  |
| 3        | Orange     | 🔵 Blue    |
| 4        | Watermelon | 🔴 Red     |

---

## Quick Reference (Copy-Paste Commands)

```powershell
# Full setup → run in one go (PowerShell)
cd "C:\Users\lakshya sharma\Desktop\cvrelated\fuitdetecteion scarch\fuitdetecteion scarch"
.\fruitenv\Scripts\Activate.ps1
python camera.py
```

```cmd
# Full setup → run in one go (Command Prompt)
cd "C:\Users\lakshya sharma\Desktop\cvrelated\fuitdetecteion scarch\fuitdetecteion scarch"
fruitenv\Scripts\activate.bat
python camera.py
```

---

## Model Training Info (Reference)

The model was trained using YOLOv8 nano segmentation (`yolov8n-seg.pt`) on a custom dataset of 5 fruit classes. Training configuration is in `fruit_final2.yaml`. The best weights are stored at:

```
runs/segment/fruit_seg_v33/weights/best.pt
```

To retrain or fine-tune:
```powershell
yolo segment train model=yolov8n-seg.pt data=fruit_final2.yaml epochs=100 imgsz=640
```

---

*Last updated: April 14, 2026*
