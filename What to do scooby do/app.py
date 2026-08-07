from flask import Flask, Response, render_template, jsonify
from ultralytics import YOLO
from picamera2 import Picamera2
import cv2
import numpy as np
import time
import sqlite3
import re

app = Flask(__name__)

# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION — Tune these on-site for your LED display
# ═══════════════════════════════════════════════════════════════

# HSV range for BLUE LED digits (your weighing machine)
# Covers cyan-to-blue hues with decent saturation and brightness
LED_HSV_LOWER = np.array([90, 80, 100])
LED_HSV_UPPER = np.array([140, 255, 255])

# Brightness threshold for the blue channel isolation
LED_BLUE_THRESH = 130

# Minimum contour area to be considered a digit (filters noise)
MIN_DIGIT_AREA = 80

# Frame split ratios
FRUIT_REGION_RATIO = 0.6        # top 60% of frame → YOLO
WEIGHT_REGION_Y_START = 0.65    # bottom 35% of frame → weight
WEIGHT_REGION_X_END = 0.5       # left 50% of frame → weight

# ═══════════════════════════════════════════════════════════════
#  LOAD MODEL & CAMERA
# ═══════════════════════════════════════════════════════════════

model = YOLO("best.pt")

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()

time.sleep(2)

latest_counts = {}
latest_weight = "0"


# ═══════════════════════════════════════════════════════════════
#  FRUIT DETECTION (YOLO)
# ═══════════════════════════════════════════════════════════════

def get_counts(results, names):
    """Count detected fruits by class name."""
    counts = {}
    boxes = results[0].boxes

    if boxes is not None:
        for cls in boxes.cls:
            name = names[int(cls)]
            counts[name] = counts.get(name, 0) + 1

    return counts


# ═══════════════════════════════════════════════════════════════
#  7-SEGMENT DIGIT RECOGNITION (LED DISPLAY)
# ═══════════════════════════════════════════════════════════════

# 7-segment layout:
#   _        segment 0 (top)
#  |_|       segments 1,2 (top-left, top-right), 3 (middle)
#  |_|       segments 4,5 (bottom-left, bottom-right), 6 (bottom)
#
# Each digit maps to which segments are ON (1) or OFF (0):
#              [top, top-left, top-right, mid, bot-left, bot-right, bot]
SEVEN_SEG = {
    (1, 1, 1, 0, 1, 1, 1): 0,
    (0, 0, 1, 0, 0, 1, 0): 1,
    (1, 0, 1, 1, 1, 0, 1): 2,
    (1, 0, 1, 1, 0, 1, 1): 3,
    (0, 1, 1, 1, 0, 1, 0): 4,
    (1, 1, 0, 1, 0, 1, 1): 5,
    (1, 1, 0, 1, 1, 1, 1): 6,
    (1, 0, 1, 0, 0, 1, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 1, 0, 1, 1): 9,
}


def classify_digit(roi):
    """
    Given a binary ROI of a single 7-segment digit,
    determine which segments are lit and return the closest digit (0-9).
    Uses fuzzy matching — picks the digit with fewest mismatched segments.
    """
    h, w = roi.shape
    if h < 8 or w < 4:
        return None

    # Define 7 segment regions as (y_start%, y_end%, x_start%, x_end%)
    # Widened zones slightly to be more forgiving of alignment
    segments = [
        (0.00, 0.20, 0.15, 0.85),   # 0: top
        (0.05, 0.48, 0.00, 0.35),   # 1: top-left
        (0.05, 0.48, 0.65, 1.00),   # 2: top-right
        (0.35, 0.65, 0.15, 0.85),   # 3: middle
        (0.52, 0.95, 0.00, 0.35),   # 4: bottom-left
        (0.52, 0.95, 0.65, 1.00),   # 5: bottom-right
        (0.80, 1.00, 0.15, 0.85),   # 6: bottom
    ]

    on_off = []
    for (y1f, y2f, x1f, x2f) in segments:
        y1, y2 = int(y1f * h), int(y2f * h)
        x1, x2 = int(x1f * w), int(x2f * w)

        # Clamp to valid range
        y1, y2 = max(0, y1), min(h, y2)
        x1, x2 = max(0, x1), min(w, x2)

        region = roi[y1:y2, x1:x2]
        if region.size == 0:
            on_off.append(0)
            continue

        fill = np.mean(region) / 255.0
        on_off.append(1 if fill > 0.20 else 0)

    key = tuple(on_off)

    # Exact match first
    if key in SEVEN_SEG:
        return SEVEN_SEG[key]

    # Fuzzy match — find digit with minimum Hamming distance
    best_digit = None
    best_dist = 999
    for pattern, digit in SEVEN_SEG.items():
        dist = sum(a != b for a, b in zip(key, pattern))
        if dist < best_dist:
            best_dist = dist
            best_digit = digit

    # Accept if at most 2 segments mismatch (out of 7)
    if best_dist <= 2:
        return best_digit

    return None


def read_weight(frame):
    """
    Read weight digits from an LED display crop using
    blue-channel isolation + 7-segment contour analysis.
    """
    # --- Method 1: HSV masking for blue LEDs ---
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.inRange(hsv, LED_HSV_LOWER, LED_HSV_UPPER)

    # --- Method 2: Blue channel isolation ---
    # For blue LEDs, the B channel is dominant
    b_channel = frame[:, :, 0]  # OpenCV BGR → index 0 is Blue
    _, blue_mask = cv2.threshold(b_channel, LED_BLUE_THRESH, 255, cv2.THRESH_BINARY)

    # Combine both methods
    combined = cv2.bitwise_or(hsv_mask, blue_mask)

    # Light morphological cleanup — don't over-dilate or digits merge
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    combined = cv2.dilate(combined, kernel_small, iterations=1)
    combined = cv2.erode(combined, kernel_small, iterations=1)

    # Close small gaps in digit segments
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close)

    # Find contours
    contours, _ = cv2.findContours(
        combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return "0"

    # Get the overall height of the tallest contour for reference
    max_h = max(cv2.boundingRect(c)[3] for c in contours)

    # Filter contours — keep only digit-sized ones
    digit_boxes = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch

        if area < MIN_DIGIT_AREA:
            continue

        # Height filter: digit should be at least 30% of tallest
        if ch < max_h * 0.3:
            continue

        # Aspect ratio: allow wide digits ("0") and narrow ("1")
        aspect = ch / max(cw, 1)
        if aspect < 0.5 or aspect > 8.0:
            continue

        digit_boxes.append((x, y, cw, ch))

    if not digit_boxes:
        return "0"

    # Sort left-to-right
    digit_boxes.sort(key=lambda b: b[0])

    # Merge overlapping boxes (keep the larger one)
    filtered = [digit_boxes[0]]
    for box in digit_boxes[1:]:
        prev = filtered[-1]
        # Skip if overlapping with previous
        if box[0] < prev[0] + prev[2] * 0.6:
            # Keep whichever is taller
            if box[3] > prev[3]:
                filtered[-1] = box
            continue
        filtered.append(box)

    # Classify each digit
    digits = []
    for (x, y, cw, ch) in filtered:
        # Add small padding around digit ROI
        pad = 2
        y1 = max(0, y - pad)
        y2 = min(combined.shape[0], y + ch + pad)
        x1 = max(0, x - pad)
        x2 = min(combined.shape[1], x + cw + pad)

        roi = combined[y1:y2, x1:x2]
        digit = classify_digit(roi)

        if digit is not None:
            digits.append(str(digit))

    result = "".join(digits)
    return result if result else "0"


# ═══════════════════════════════════════════════════════════════
#  BILLING (SQLite)
# ═══════════════════════════════════════════════════════════════

def calculate_bill(counts, weight):
    conn = sqlite3.connect("store.db")
    c = conn.cursor()

    bill = []
    total_sum = 0

    try:
        weight_val = float(weight) / 100  # convert 34 → 0.34 kg
    except:
        weight_val = 0

    for item in counts:
        c.execute("SELECT price FROM prices WHERE name=?", (item,))
        result = c.fetchone()

        if result:
            price = result[0]
            total = price * weight_val
            total_sum += total

            bill.append({
                "item": item,
                "weight": weight_val,
                "price": price,
                "total": round(total, 2)
            })

    conn.close()
    return bill, round(total_sum, 2)


# ═══════════════════════════════════════════════════════════════
#  VIDEO STREAM
# ═══════════════════════════════════════════════════════════════

def generate_frames():
    global latest_counts, latest_weight

    while True:
        frame = picam2.capture_array()
        frame = frame[:, :, :3]

        h, w, _ = frame.shape

        # ─── SPLIT FRAME ────────────────────────────
        top = frame[0:int(h * FRUIT_REGION_RATIO), :]          # fruit area
        bottom = frame[int(h * WEIGHT_REGION_Y_START):h,
                       0:int(w * WEIGHT_REGION_X_END)]          # LED display

        # ─── YOLO (fruit detection) ──────────────────
        results = model(top)
        latest_counts = get_counts(results, model.names)

        # ─── LED DIGIT READING (weight) ──────────────
        latest_weight = read_weight(bottom)

        # ─── DRAW OUTPUT ─────────────────────────────
        annotated_top = results[0].plot()

        combined = frame.copy()
        combined[0:int(h * FRUIT_REGION_RATIO), :] = annotated_top

        # Show weight on frame
        try:
            display_weight = str(int(latest_weight) / 100)
        except:
            display_weight = "0.0"

        cv2.putText(combined, f"Weight: {display_weight} kg",
                    (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

        # Fix color (Picamera2 gives RGB, OpenCV expects BGR)
        combined = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)

        _, buffer = cv2.imencode('.jpg', combined)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# ═══════════════════════════════════════════════════════════════
#  FLASK ROUTES
# ═══════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/data')
def data():
    bill, total = calculate_bill(latest_counts, latest_weight)

    return jsonify({
        "bill": bill,
        "weight": str(int(latest_weight)/100),
        "total": total
    })


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)