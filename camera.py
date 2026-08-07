# camera.py  —  Fruit Detection + OCR Weight Reader + Billing Panel
import cv2
import numpy as np
import time
import argparse
import re
import threading
from ultralytics import YOLO
from collections import Counter
import easyocr

# ─── PRICES PER KG (INR) ──────────────────────────────────────────────────────
PRICES = {
    "Apple":      120,
    "Banana":      40,
    "Mango":       80,
    "Orange":      60,
    "Watermelon":  25,
}

FRUIT_COLORS = {
    0: (0, 255, 0),      # Apple      → Green
    1: (0, 255, 255),    # Banana     → Yellow
    2: (255, 165, 0),    # Mango      → Orange
    3: (0, 128, 255),    # Orange     → Blue
    4: (0, 0, 255),      # Watermelon → Red
}
NAMES = {0: "Apple", 1: "Banana", 2: "Mango", 3: "Orange", 4: "Watermelon"}

WIN_NAME            = "Fruit Detection + Billing"
WEIGHT_ZONE_H_RATIO = 0.18   # bottom 18% of frame = weight zone
BILL_PANEL_RATIO    = 0.22   # right 22% of frame  = billing panel


# ─── CAMERA ───────────────────────────────────────────────────────────────────
def find_usb_camera(preferred_index=None):
    if preferred_index is not None:
        cap = cv2.VideoCapture(preferred_index, cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.release()
            print(f"[INFO] Using forced camera index {preferred_index}")
            return preferred_index
        print(f"[ERROR] Camera index {preferred_index} not available.")
        return None
    for idx in [1, 2, 3, 0]:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                label = "USB" if idx > 0 else "built-in"
                print(f"[INFO] Found {label} camera at index {idx}")
                return idx
    return None


# ─── OCR ──────────────────────────────────────────────────────────────────────

# Green LED display HSV range  (hue 35-90°, reasonable sat/val)
_GREEN_LO = np.array([35,  40,  40], dtype=np.uint8)
_GREEN_HI = np.array([90, 255, 255], dtype=np.uint8)


def _crop_to_display(roi):
    """
    Find the largest green region in the ROI (the LED display face)
    and return a tightly-cropped version of it.
    Falls back to the full ROI if nothing is found.
    """
    hsv   = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask  = cv2.inRange(hsv, _GREEN_LO, _GREEN_HI)
    # fill small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return roi, False

    largest = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(largest) < 300:   # too small → noise
        return roi, False

    x, y, bw, bh = cv2.boundingRect(largest)
    pad = 4
    x  = max(0, x  - pad);  y  = max(0, y  - pad)
    bw = min(roi.shape[1] - x, bw + 2 * pad)
    bh = min(roi.shape[0] - y, bh + 2 * pad)
    return roi[y:y + bh, x:x + bw], True


def preprocess_for_ocr(roi):
    """
    Specialised for green LED / LCD display with black digit segments.

    Pipeline:
      1. Auto-crop to the green display face via HSV masking.
      2. Pull the GREEN channel  (high = green background, low = black digit).
      3. Invert  →  digits become bright white on dark background.
      4. Upscale 3× for better OCR resolution.
      5. CLAHE  →  equalise brightness across the panel.
      6. Otsu threshold  →  clean binary image.
      7. Morphological close  →  fill gaps in 7-segment strokes.
    """
    cropped, found = _crop_to_display(roi)

    # Green channel carries the most information for this display type
    green_ch = cropped[:, :, 1]          # BGR index 1 = green

    # Invert: background (bright green) → dark, digits (black) → bright
    inverted = cv2.bitwise_not(green_ch)

    # Upscale
    up = cv2.resize(inverted,
                    (inverted.shape[1] * 3, inverted.shape[0] * 3),
                    interpolation=cv2.INTER_CUBIC)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(6, 6))
    eq    = clahe.apply(up)

    # Threshold
    _, thresh = cv2.threshold(eq, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Close small gaps inside digit strokes
    k       = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k)

    return cleaned


def extract_weight_kg(ocr_results):
    """Parse EasyOCR result list → float kg. Handles kg / g / bare number."""
    # flatten all detected strings
    full = " ".join(r[1] for r in ocr_results).lower().strip()
    # fix common 7-segment OCR confusions before parsing
    full = full.replace('o', '0').replace('O', '0')
    full = full.replace('l', '1').replace('I', '1').replace('i', '1')
    full = full.replace('s', '5').replace('S', '5')
    full = full.replace('b', '6').replace('B', '8')

    m = re.search(r'(\d+\.?\d*)\s*kg', full)
    if m:
        return round(float(m.group(1)), 3)
    m = re.search(r'(\d+\.?\d*)\s*g\b', full)
    if m:
        return round(float(m.group(1)) / 1000.0, 3)
    # bare decimal  (e.g. "0.450" or "1.234")
    m = re.search(r'\b(\d+\.\d+)\b', full)
    if m:
        return round(float(m.group(1)), 3)
    # bare integer — treat > 50 as grams
    m = re.search(r'\b(\d{1,4})\b', full)
    if m:
        val = float(m.group(1))
        return round(val / 1000.0, 3) if val > 50 else round(val, 3)
    return None


class OCRWorker:
    def __init__(self):
        print("[INFO] Loading EasyOCR (first load ~15 s) …")
        self.reader    = easyocr.Reader(['en'], gpu=False, verbose=False)
        print("[INFO] EasyOCR ready.")
        self._lock      = threading.Lock()
        self._roi       = None
        self._weight    = None
        self._raw_text  = ""
        self._debug_img = None   # preprocessed image for on-screen debug
        self._running   = True
        threading.Thread(target=self._loop, daemon=True).start()

    def update_roi(self, roi):
        with self._lock:
            self._roi = roi.copy()

    def get_result(self):
        with self._lock:
            return self._weight, self._raw_text, self._debug_img

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            roi = None
            with self._lock:
                if self._roi is not None:
                    roi, self._roi = self._roi.copy(), None
            if roi is not None:
                proc    = preprocess_for_ocr(roi)
                results = self.reader.readtext(
                    proc,
                    allowlist='0123456789.',   # digits + decimal only
                    detail=1,
                    paragraph=False,
                    contrast_ths=0.1,
                    adjust_contrast=0.8,
                    text_threshold=0.6,
                    low_text=0.3,
                )
                wt  = extract_weight_kg(results)
                raw = " ".join(r[1] for r in results).strip()
                # shrink debug image to a fixed thumbnail height
                dbg = cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR)
                th  = 80
                if dbg.shape[0] > 0:
                    tw  = int(dbg.shape[1] * th / dbg.shape[0])
                    dbg = cv2.resize(dbg, (max(tw, 1), th))
                with self._lock:
                    self._weight    = wt
                    self._raw_text  = raw
                    self._debug_img = dbg
            else:
                time.sleep(0.04)


# ─── DRAWING — all sizes derived from (h, w) ──────────────────────────────────
def s(h, base=720):
    """Scale factor relative to 720 p reference height."""
    return h / base


def draw_weight_zone(frame, zone_y, bill_w, weight_kg, raw_text, debug_img=None):
    h, w = frame.shape[:2]
    sc   = s(h)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, zone_y), (w - bill_w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.rectangle(frame, (6, zone_y + 6),
                  (w - bill_w - 6, h - 6), (0, 220, 255), 2)

    fs_label = 0.55 * sc
    fs_raw   = 0.44 * sc
    fs_wt    = 1.5  * sc
    th_label = int(22 * sc)
    th_raw   = int(18 * sc)

    cv2.putText(frame,
                "WEIGHT ZONE  —  Place scale display here",
                (18, zone_y + th_label + 6),
                cv2.FONT_HERSHEY_SIMPLEX, fs_label, (0, 220, 255),
                max(1, int(sc * 1.5)))

    if raw_text:
        cv2.putText(frame, f"OCR raw: \"{raw_text[:50]}\"",
                    (18, zone_y + th_label + th_raw + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, fs_raw, (160, 160, 160), 1)

    # ── Debug thumbnail (what OCR actually sees) ──
    if debug_img is not None:
        try:
            dh, dw = debug_img.shape[:2]
            # scale thumbnail to fit zone height with margin
            max_th = max(1, h - zone_y - int(50 * sc))
            if dh > max_th:
                dw = int(dw * max_th / dh)
                dh = max_th
                debug_img = cv2.resize(debug_img, (dw, dh))
            # place it on the right side of the weight zone
            tx = w - bill_w - dw - 12
            ty = zone_y + 8
            if tx > 0 and ty + dh < h:
                frame[ty:ty + dh, tx:tx + dw] = debug_img
                # label
                cv2.putText(frame, "OCR view",
                            (tx, ty - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38 * sc,
                            (120, 120, 120), 1)
                cv2.rectangle(frame, (tx - 1, ty - 1),
                              (tx + dw + 1, ty + dh + 1),
                              (80, 80, 80), 1)
        except Exception:
            pass

    wt_str = f"{weight_kg:.3f} kg" if weight_kg is not None else "-- kg"
    wt_col = (0, 255, 180)        if weight_kg is not None else (80, 80, 80)
    wt_thk = max(2, int(3 * sc))
    cv2.putText(frame, wt_str, (18, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, fs_wt, wt_col, wt_thk)


def draw_billing_panel(frame, bill_items, current_fruit, weight_kg):
    h, w   = frame.shape[:2]
    sc     = s(h)
    bill_w = int(w * BILL_PANEL_RATIO)
    px     = w - bill_w

    # Dark background
    overlay = frame.copy()
    cv2.rectangle(overlay, (px, 0), (w, h), (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
    cv2.line(frame, (px, 0), (px, h), (180, 180, 180), 1)

    pad  = int(10 * sc)
    lx   = px + pad

    # Header bar
    hdr_h = int(44 * sc)
    cv2.rectangle(frame, (px, 0), (w, hdr_h), (28, 28, 100), -1)
    cv2.putText(frame, " BILLING", (lx, int(30 * sc)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.80 * sc, (255, 255, 255),
                max(1, int(2 * sc)))

    y = hdr_h + int(14 * sc)

    # ── Current item ──
    cv2.putText(frame, "Current Item", (lx, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50 * sc, (160, 210, 255),
                max(1, int(sc)))
    y += int(24 * sc)

    if current_fruit and weight_kg:
        price    = PRICES.get(current_fruit, 0)
        subtotal = round(weight_kg * price, 2)
        col = FRUIT_COLORS.get(
            [k for k, v in NAMES.items() if v == current_fruit][0],
            (255, 255, 255))
        cv2.putText(frame, current_fruit, (lx, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75 * sc, col,
                    max(1, int(2 * sc)))
        y += int(26 * sc)
        cv2.putText(frame,
                    f"  {weight_kg:.3f} kg x Rs{price}",
                    (lx, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50 * sc, (200, 200, 200),
                    max(1, int(sc)))
        y += int(24 * sc)
        cv2.putText(frame, f"  = Rs {subtotal:.2f}", (lx, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62 * sc, (0, 255, 180),
                    max(1, int(2 * sc)))
    else:
        cv2.putText(frame, "No fruit / weight", (lx, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48 * sc, (80, 80, 80),
                    max(1, int(sc)))
    y += int(20 * sc)

    cv2.putText(frame, "[A] Add   [C] Clear   [Q] Quit",
                (lx, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38 * sc, (100, 180, 100),
                max(1, int(sc)))
    y += int(14 * sc)

    # Divider
    cv2.line(frame, (px + pad, y), (w - pad, y), (70, 70, 70), 1)
    y += int(14 * sc)

    # ── Bill list ──
    cv2.putText(frame, "Bill", (lx, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60 * sc, (255, 220, 80),
                max(1, int(2 * sc)))
    y += int(22 * sc)

    total_bar_h = int(52 * sc)
    max_y       = h - total_bar_h - int(8 * sc)

    if not bill_items:
        cv2.putText(frame, "  (empty)", (lx, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45 * sc, (70, 70, 70),
                    max(1, int(sc)))
    else:
        for i, (fname, wt, ppu, sub) in enumerate(bill_items):
            if y + int(36 * sc) > max_y:
                cv2.putText(frame,
                            f"  ...+{len(bill_items)-i} more",
                            (lx, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40 * sc,
                            (120, 120, 120), max(1, int(sc)))
                break
            col = FRUIT_COLORS.get(
                [k for k, v in NAMES.items() if v == fname][0],
                (255, 255, 255))
            cv2.putText(frame,
                        f"{i+1}. {fname[:8]}  {wt:.3f}kg",
                        (lx, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46 * sc, col,
                        max(1, int(sc)))
            y += int(20 * sc)
            cv2.putText(frame,
                        f"   Rs{ppu}/kg = Rs{sub:.2f}",
                        (lx, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40 * sc, (190, 190, 190),
                        max(1, int(sc)))
            y += int(20 * sc)

    # ── Grand total bar ──
    cv2.rectangle(frame, (px, h - total_bar_h), (w, h), (25, 55, 25), -1)
    cv2.line(frame, (px, h - total_bar_h), (w, h - total_bar_h),
             (80, 180, 80), 1)
    total = sum(sub for _, _, _, sub in bill_items)
    cv2.putText(frame, f"TOTAL: Rs {total:.2f}",
                (lx, h - int(14 * sc)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72 * sc, (0, 255, 100),
                max(1, int(2 * sc)))


def draw_fruit_panel(frame, fruit_counter, bill_w):
    h, w = frame.shape[:2]
    sc   = s(h)
    n    = len(fruit_counter)
    ph   = int((35 + (n + 1) * 28) * sc)
    pw   = int(210 * sc)

    cv2.rectangle(frame, (0, 0), (pw, ph), (0, 0, 0), -1)
    cv2.rectangle(frame, (0, 0), (pw, ph), (220, 220, 220), 1)
    cv2.putText(frame, "Fruit Count", (int(10 * sc), int(26 * sc)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68 * sc, (255, 255, 255),
                max(1, int(2 * sc)))
    py = int(52 * sc)
    for fruit, count in sorted(fruit_counter.items()):
        col = FRUIT_COLORS[[k for k, v in NAMES.items() if v == fruit][0]]
        cv2.putText(frame, f"  {fruit}: {count}",
                    (int(10 * sc), py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58 * sc, col,
                    max(1, int(sc + 0.5)))
        py += int(28 * sc)
    total_fruits = sum(fruit_counter.values())
    cv2.putText(frame, f"  Total: {total_fruits}",
                (int(10 * sc), py + int(4 * sc)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60 * sc, (255, 255, 255),
                max(1, int(sc + 0.5)))


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera",     type=int,   default=None)
    parser.add_argument("--conf",       type=float, default=0.4)
    parser.add_argument("--width",      type=int,   default=1920)
    parser.add_argument("--height",     type=int,   default=1080)
    parser.add_argument("--fullscreen", action="store_true", default=True,
                        help="Launch in fullscreen (default: on)")
    parser.add_argument("--no-fullscreen", dest="fullscreen",
                        action="store_false")
    args = parser.parse_args()

    model_path = "runs/segment/fruit_seg_v33/weights/best.pt"
    model      = YOLO(model_path)
    print(f"[INFO] YOLO loaded — {model_path}")

    ocr_worker = OCRWorker()

    cam_index = find_usb_camera(args.camera)
    if cam_index is None:
        print("[ERROR] No camera found.")
        return

    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)

    # Try requested resolution; camera will give its best supported size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera resolution: {actual_w}x{actual_h}")

    # Display at 1280x720 minimum so UI is readable on any camera
    DISP_W = max(actual_w, 1280)
    DISP_H = max(actual_h, 720)
    if actual_w < DISP_W or actual_h < DISP_H:
        print(f"[INFO] Upscaling display to {DISP_W}x{DISP_H}")

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {cam_index}")
        return

    # ── Window setup ──
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    if args.fullscreen:
        cv2.setWindowProperty(WIN_NAME,
                              cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(WIN_NAME, actual_w, actual_h)

    print(f"[INFO] Camera {cam_index} started.")
    print("[INFO] A = add item   C = clear bill   Q = quit   F = toggle fullscreen")

    bill_items   = []
    ocr_frame_n  = 0
    OCR_EVERY    = 8
    weight_kg    = None
    raw_ocr_text = ""
    debug_img    = None
    fps = 0.0; frame_cnt = 0; fps_timer = time.time()
    fullscreen   = args.fullscreen

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Camera read failed")
            break

        # Upscale to display size if camera is lower resolution
        if frame.shape[1] < DISP_W or frame.shape[0] < DISP_H:
            frame = cv2.resize(frame, (DISP_W, DISP_H),
                               interpolation=cv2.INTER_LINEAR)

        h, w     = frame.shape[:2]
        sc       = s(h)
        bill_w   = int(w * BILL_PANEL_RATIO)
        zone_y   = int(h * (1 - WEIGHT_ZONE_H_RATIO))

        # ── YOLO ──
        results = model(frame, conf=args.conf, verbose=False)
        result  = results[0]

        fruit_counter = Counter()
        overlay       = frame.copy()
        largest_fruit = None
        largest_area  = 0

        if result.masks is not None:
            for mask, cls, conf, box in zip(result.masks.xy,
                                            result.boxes.cls,
                                            result.boxes.conf,
                                            result.boxes.xyxy):
                cls_id = int(cls)
                color  = FRUIT_COLORS.get(cls_id, (255, 255, 255))
                label  = NAMES.get(cls_id, "Unknown")
                fruit_counter[label] += 1

                pts = np.array(mask, dtype=np.int32)
                if len(pts) > 0:
                    cv2.fillPoly(overlay, [pts], color)

                x1, y1, x2, y2 = map(int, box)
                area = (x2 - x1) * (y2 - y1)
                if area > largest_area:
                    largest_area, largest_fruit = area, label

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, max(1, int(2 * sc)))
                text = f"{label} {conf:.2f}"
                fs   = 0.60 * sc
                tk   = max(1, int(2 * sc))
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, tk)
                cv2.rectangle(frame,
                              (x1, y1 - th - int(10 * sc)),
                              (x1 + tw + int(4 * sc), y1),
                              color, -1)
                cv2.putText(frame, text,
                            (x1 + int(2 * sc), y1 - int(5 * sc)),
                            cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), tk)

        frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

        # ── OCR ──
        ocr_frame_n += 1
        if ocr_frame_n % OCR_EVERY == 0:
            ocr_worker.update_roi(frame[zone_y:h, 0:w - bill_w])
        weight_kg, raw_ocr_text, debug_img = ocr_worker.get_result()

        # ── Panels ──
        draw_weight_zone(frame, zone_y, bill_w, weight_kg, raw_ocr_text, debug_img)
        draw_fruit_panel(frame, fruit_counter, bill_w)
        draw_billing_panel(frame, bill_items, largest_fruit, weight_kg)

        # ── FPS ──
        frame_cnt += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            fps = frame_cnt / elapsed
            frame_cnt = 0
            fps_timer = time.time()
        fps_text = f"FPS: {fps:.1f}"
        fs_fps   = 0.65 * sc
        tk_fps   = max(1, int(2 * sc))
        (fw, fh), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, fs_fps, tk_fps)
        fx = w - bill_w - fw - int(18 * sc)
        cv2.rectangle(frame, (fx - 4, 0), (fx + fw + 8, fh + int(14 * sc)), (0,0,0), -1)
        cv2.putText(frame, fps_text, (fx, fh + int(8 * sc)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs_fps, (0, 255, 0), tk_fps)

        cv2.imshow(WIN_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key in (ord('a'), ord('A')):
            if largest_fruit and weight_kg and weight_kg > 0:
                price    = PRICES.get(largest_fruit, 0)
                subtotal = round(weight_kg * price, 2)
                bill_items.append((largest_fruit, weight_kg, price, subtotal))
                print(f"[BILL] + {largest_fruit}  {weight_kg:.3f} kg  "
                      f"Rs{price}/kg  = Rs{subtotal:.2f}")
            else:
                print("[BILL] Need a detected fruit AND a weight reading first.")
        elif key in (ord('c'), ord('C')):
            bill_items.clear()
            print("[BILL] Cleared.")
        elif key in (ord('f'), ord('F')):
            fullscreen = not fullscreen
            cv2.setWindowProperty(
                WIN_NAME, cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)

    ocr_worker.stop()
    cap.release()
    cv2.destroyAllWindows()

    if bill_items:
        print("\n── Final Bill ──────────────────────────────")
        for i, (f, wt, ppu, sub) in enumerate(bill_items, 1):
            print(f"  {i}. {f:<12}  {wt:.3f} kg  x  Rs{ppu}/kg  =  Rs{sub:.2f}")
        print(f"  {'TOTAL':>38}  Rs{sum(s for _,_,_,s in bill_items):.2f}")
        print("────────────────────────────────────────────")
    print("[INFO] Session ended.")


if __name__ == '__main__':
    main()
