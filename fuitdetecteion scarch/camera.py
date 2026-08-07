# camera.py
import cv2
import numpy as np
from collections import Counter, deque
from ultralytics import YOLO
import easyocr
import re
import threading
import time


# ──────────────────────────────────────────────────────────────
# CAMERA — fixed at USB index 0 (do not change index)
# ──────────────────────────────────────────────────────────────
def open_usb_camera():
    backends = [
        ("DSHOW", cv2.CAP_DSHOW),
        ("MSMF",  cv2.CAP_MSMF),
        ("ANY",   cv2.CAP_ANY),
    ]
    for bname, backend in backends:
        print(f"Trying index 0 with {bname}...")
        try:
            cap = cv2.VideoCapture(0, backend)
        except Exception as e:
            print(f"  Exception: {e}")
            continue
        if not cap.isOpened():
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        frame = None
        for _ in range(20):
            ret, f = cap.read()
            if ret and f is not None and f.mean() > 5:
                frame = f
                break

        if frame is None:
            cap.release()
            continue

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera ready: index 0 ({bname}), {w}x{h}")
        return cap

    print("Cannot open camera at index 0.")
    return None


# ──────────────────────────────────────────────────────────────
# FRUIT VALIDATION
# ──────────────────────────────────────────────────────────────
def is_skin_dominant(bgr_roi, threshold=0.35):
    if bgr_roi.size == 0:
        return False
    hsv  = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    lo   = np.array([0,  30,  50], dtype=np.uint8)
    hi   = np.array([22, 170, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lo, hi)
    return (np.count_nonzero(mask) / mask.size) > threshold


def is_valid_fruit(box, mask_pts, frame_shape):
    fh, fw = frame_shape[:2]
    x1, y1, x2, y2 = box
    box_area = (x2 - x1) * (y2 - y1)
    if box_area > fh * fw * 0.40:
        return False
    if len(mask_pts) >= 4:
        hull     = cv2.convexHull(mask_pts)
        solidity = cv2.contourArea(mask_pts) / max(cv2.contourArea(hull), 1)
        if solidity < 0.35:
            return False
    return True


# ──────────────────────────────────────────────────────────────
# LCD DISPLAY DETECTION
# ──────────────────────────────────────────────────────────────
def find_lcd_display(scan_zone):
    hsv = cv2.cvtColor(scan_zone, cv2.COLOR_BGR2HSV)
    v   = hsv[:, :, 2]
    best_rect, best_score = None, 0
    for thresh_val in (200, 170, 140, 110):
        _, bright = cv2.threshold(v, thresh_val, 255, cv2.THRESH_BINARY)
        k  = cv2.getStructuringElement(cv2.MORPH_RECT, (22, 8))
        k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (4,  4))
        bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, k,  iterations=3)
        bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN,  k2, iterations=1)
        contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area   = w * h
            aspect = w / max(h, 1)
            if area < 800 or aspect < 1.2 or h < 10:
                continue
            if area > best_score:
                best_score, best_rect = area, (x, y, w, h)

    if best_rect is None:
        return None, None
    x, y, w, h = best_rect
    pad = 12
    x1 = max(0, x - pad);  y1 = max(0, y - pad)
    x2 = min(scan_zone.shape[1], x + w + pad)
    y2 = min(scan_zone.shape[0], y + h + pad)
    return scan_zone[y1:y2, x1:x2].copy(), (x1, y1, x2 - x1, y2 - y1)


# ──────────────────────────────────────────────────────────────
# OCR — lean preprocessing, best 4 variants only
# ──────────────────────────────────────────────────────────────
UPSCALE = 6

def _make_ocr_variants(img):
    h, w = img.shape[:2]
    up   = cv2.resize(img, (w * UPSCALE, h * UPSCALE), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)

    # Variant 1: CLAHE + Otsu
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4, 4))
    eq    = clahe.apply(gray)
    _, v1 = cv2.threshold(eq,   0, 255, cv2.THRESH_BINARY     + cv2.THRESH_OTSU)
    _, v2 = cv2.threshold(eq,   0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Variant 2: Bilateral + Otsu
    bil   = cv2.bilateralFilter(gray, 9, 75, 75)
    _, v3 = cv2.threshold(bil,  0, 255, cv2.THRESH_BINARY     + cv2.THRESH_OTSU)
    _, v4 = cv2.threshold(bil,  0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Dilate strokes for thin 7-seg displays
    dk = np.ones((2, 2), np.uint8)
    variants = [v1, v2, v3, v4,
                cv2.dilate(v1, dk), cv2.dilate(v2, dk)]

    return [cv2.cvtColor(v, cv2.COLOR_GRAY2BGR) for v in variants]


def _is_valid_weight(text):
    """Accept only plausible scale readings: 1–9999 g, up to one decimal place."""
    try:
        val = float(text)
    except ValueError:
        return False
    if val <= 0 or val > 9999:
        return False
    # Must match NNN or NNN.N / NN.NN — not just a single digit
    if not re.fullmatch(r'\d{1,4}(\.\d{1,2})?', text):
        return False
    return True


def run_ocr(reader, display_img):
    variants   = _make_ocr_variants(display_img)
    candidates = []
    for variant in variants:
        try:
            raw = reader.readtext(
                variant,
                allowlist='0123456789.',
                detail=1, paragraph=False, batch_size=1,
                text_threshold=0.25, low_text=0.15,
                link_threshold=0.3,
                mag_ratio=1.0,
                decoder='greedy',          # beamsearch causes overflow warnings
                width_ths=0.8, height_ths=0.6,
            )
            for bbox, text, conf in raw:
                cleaned = re.sub(r'[^0-9.]', '', text)
                if not _is_valid_weight(cleaned):
                    continue
                candidates.append((bbox, cleaned, conf))
        except Exception:
            pass

    if not candidates:
        return "", []

    def score(item):
        _, text, conf = item
        s = conf
        if '.' in text:
            s += 0.08
        if 2 <= len(text.replace('.', '')) <= 5:
            s += 0.12
        return s

    candidates.sort(key=score, reverse=True)
    seen, deduped = set(), []
    for b, t, c in candidates:
        if t not in seen:
            seen.add(t)
            deduped.append((b, t, c))

    best_text  = deduped[0][1]
    detections = [(np.array(b, dtype=np.float32) / UPSCALE, t, c)
                  for b, t, c in deduped[:3]]
    return best_text, detections


def stable_number(history):
    non_empty = [v for v in history if v]
    return max(set(non_empty), key=non_empty.count) if non_empty else ""


# ──────────────────────────────────────────────────────────────
# BACKGROUND OCR WORKER — never blocks the display loop
# ──────────────────────────────────────────────────────────────
class OcrWorker:
    """
    Producer/consumer: main loop pushes scan zones here;
    worker thread processes them; main loop reads latest result.
    OCR never touches the display loop.
    """
    def __init__(self):
        self._lock    = threading.Lock()
        self._pending = None          # next frame to process
        self._result  = ("", [], None)  # (text, dets, display_rect)
        self._reader  = None
        self._ready   = False
        self._running = True
        threading.Thread(target=self._load_and_run, daemon=True).start()

    def _load_and_run(self):
        print("EasyOCR loading in background...")
        self._reader = easyocr.Reader(['en'], gpu=True, verbose=False)
        self._ready  = True
        print("EasyOCR ready.")
        while self._running:
            img = None
            with self._lock:
                if self._pending is not None:
                    img, self._pending = self._pending, None
            if img is None:
                time.sleep(0.03)
                continue
            try:
                display_crop, rect = find_lcd_display(img)
                src = display_crop if (display_crop is not None and display_crop.size > 0) else img
                text, dets = run_ocr(self._reader, src)
                with self._lock:
                    self._result = (text, dets, rect)
            except Exception:
                pass

    def submit(self, scan_zone_img):
        """Give the worker a new image (drops old pending if not yet consumed)."""
        if not self._ready:
            return
        with self._lock:
            self._pending = scan_zone_img.copy()

    def get_result(self):
        """Returns (text, detections, display_rect) — never blocks."""
        with self._lock:
            return self._result

    @property
    def is_ready(self):
        return self._ready

    def stop(self):
        self._running = False


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    cap = open_usb_camera()
    if cap is None:
        return

    model = YOLO("runs/segment/fruit_seg_v33/weights/best.pt")

    names  = {0: "Apple", 1: "Banana", 2: "Mango", 3: "Orange", 4: "Watermelon"}
    colors = {
        0: (0, 255, 0),
        1: (0, 255, 255),
        2: (255, 165, 0),
        3: (0, 128, 255),
        4: (0, 0, 255),
    }

    ocr_worker   = OcrWorker()
    ocr_history  = deque(maxlen=15)
    frame_count  = 0
    OCR_EVERY    = 10    # submit a new scan every N frames; OCR runs async

    print("Camera running — press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Camera read error — exiting")
            break
        frame_count += 1

        # ── YOLO (imgsz=640 keeps inference fast regardless of cam res) ──
        results = model(
            frame,
            imgsz=640,
            conf=0.25,
            iou=0.50,
            max_det=30,
            agnostic_nms=False,
            verbose=False,
        )
        result  = results[0]

        fruit_counter = Counter()
        overlay       = frame.copy()

        if result.masks is not None and len(result.boxes) > 0:
            for mask, cls, conf, box in zip(
                result.masks.xy,
                result.boxes.cls,
                result.boxes.conf,
                result.boxes.xyxy,
            ):
                conf_f = float(conf)
                cls_id = int(cls)
                x1, y1, x2, y2 = map(int, box)
                pts = np.array(mask, dtype=np.int32)

                if not is_valid_fruit((x1, y1, x2, y2), pts, frame.shape):
                    continue
                roi = frame[max(y1, 0):y2, max(x1, 0):x2]
                if is_skin_dominant(roi):
                    continue

                color = colors.get(cls_id, (255, 255, 255))
                label = names.get(cls_id, "Unknown")
                fruit_counter[label] += 1

                if len(pts) > 0:
                    cv2.fillPoly(overlay, [pts], color)

                if (x2 - x1) * (y2 - y1) < 400:
                    continue
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                lbl = f"{label} {conf_f:.2f}"
                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 6, y1), color, -1)
                cv2.putText(frame, lbl, (x1 + 3, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

        # ── FRUIT COUNT PANEL ─────────────────────────────────
        panel_rows = max(len(fruit_counter), 1)
        panel_h    = 40 + (panel_rows + 1) * 34 + 10
        cv2.rectangle(frame, (0, 0), (270, panel_h), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, 0), (270, panel_h), (255, 255, 255), 1)
        cv2.putText(frame, "Fruit Count", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        panel_y = 64
        for fruit, count in sorted(fruit_counter.items()):
            col = colors[next(k for k, v in names.items() if v == fruit)]
            cv2.putText(frame, f"{fruit}: {count}", (10, panel_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2)
            panel_y += 34
        total = sum(fruit_counter.values())
        cv2.putText(frame, f"Total: {total}", (10, panel_y + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # ── OCR SCAN ZONE (compact strip at bottom) ───────────
        fh, fw = frame.shape[:2]
        sz_h  = 100                        # compact strip — just enough for scale display
        sz_y1, sz_y2 = fh - sz_h, fh - 8
        sz_x1, sz_x2 = 20, fw - 20

        cv2.rectangle(frame, (sz_x1, sz_y1), (sz_x2, sz_y2), (200, 200, 200), 1)
        cv2.putText(frame, "Scale OCR zone", (sz_x1 + 4, sz_y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Submit new scan to background worker every OCR_EVERY frames
        if frame_count % OCR_EVERY == 0:
            ocr_worker.submit(frame[sz_y1:sz_y2, sz_x1:sz_x2])

        # ── READ LATEST OCR RESULT (non-blocking) ─────────────
        ocr_text, ocr_det_cache, display_rect_cache = ocr_worker.get_result()
        if ocr_text:
            # Only add if different from last to avoid spamming history
            if not ocr_history or ocr_history[-1] != ocr_text:
                ocr_history.append(ocr_text)

        # ── DRAW LCD BOX + OCR OVERLAYS ───────────────────────
        if display_rect_cache is not None:
            dx, dy, dw, dh = display_rect_cache
            fdx1, fdy1 = sz_x1 + dx, sz_y1 + dy
            cv2.rectangle(frame, (fdx1, fdy1), (fdx1 + dw, fdy1 + dh), (0, 255, 0), 2)
            cv2.putText(frame, "Display", (fdx1, fdy1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            if ocr_det_cache:
                live = frame[fdy1:fdy1 + dh, fdx1:fdx1 + dw]
                if live.size > 0:
                    for pts, text, conf in ocr_det_cache:
                        pi = np.clip(pts.astype(np.int32), 0,
                                     [live.shape[1] - 1, live.shape[0] - 1])
                        cv2.polylines(live, [pi], True, (0, 255, 255), 2)
                        x0, y0 = pi[0]
                        cv2.putText(live, f"{text}({conf:.2f})",
                                    (x0, max(y0 - 4, 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # ── WEIGHT DISPLAY ────────────────────────────────────
        weight_str   = stable_number(ocr_history)
        weight_label = f"Weight: {weight_str} g" if weight_str else "Weight: ---"
        label_y      = panel_h + 42
        (wl_w, wl_h), _ = cv2.getTextSize(weight_label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(frame, (0, label_y - wl_h - 8), (wl_w + 18, label_y + 10), (0, 0, 0), -1)
        cv2.putText(frame, weight_label, (8, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 255) if weight_str else (100, 100, 100), 2)

        if not ocr_worker.is_ready:
            cv2.putText(frame, "OCR loading...", (8, label_y + 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)

        # ── TOP-RIGHT TOTAL ───────────────────────────────────
        cv2.putText(frame, f"Fruits: {total}", (fw - 210, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Fruit Detection System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    ocr_worker.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == '__main__':
    main()
