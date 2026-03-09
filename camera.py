# camera.py
import cv2
import numpy as np
from ultralytics import YOLO
from collections import Counter

def main():
    # ─── LOAD BEST MODEL ──────────────────────────────────────────
    model = YOLO("runs/segment/fruit_seg_v33/weights/best.pt")

    # ─── CLASS NAMES AND COLORS ───────────────────────────────────
    names = {
        0: "Apple",
        1: "Banana",
        2: "Mango",
        3: "Orange",
        4: "Watermelon"
    }
    colors = {
        0: (0, 255, 0),      # Apple      → Green
        1: (0, 255, 255),    # Banana     → Yellow
        2: (255, 165, 0),    # Mango      → Orange
        3: (0, 128, 255),    # Orange     → Blue
        4: (0, 0, 255)       # Watermelon → Red
    }

    # ─── OPEN LAPTOP WEBCAM ───────────────────────────────────────
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("❌ Cannot open camera")
        return

    print("✅ Camera started — press Q to quit")
    print("✅ Model loaded — fruit_seg_v33/weights/best.pt")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Camera error")
            break

        # ─── RUN INFERENCE ────────────────────────────────────────
        results = model(frame, conf=0.4, verbose=False)
        result  = results[0]

        fruit_counter = Counter()
        overlay       = frame.copy()

        # ─── DRAW MASKS + BOXES ───────────────────────────────────
        if result.masks is not None:
            masks   = result.masks.xy
            classes = result.boxes.cls
            confs   = result.boxes.conf
            boxes   = result.boxes.xyxy

            for mask, cls, conf, box in zip(masks, classes, confs, boxes):
                cls_id = int(cls)
                color  = colors.get(cls_id, (255, 255, 255))
                label  = names.get(cls_id, "Unknown")
                fruit_counter[label] += 1

                # Draw filled mask
                pts = np.array(mask, dtype=np.int32)
                if len(pts) > 0:
                    cv2.fillPoly(overlay, [pts], color)

                # Draw bounding box
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Draw label + confidence
                text = f"{label} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
                cv2.putText(frame, text, (x1 + 2, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # ─── BLEND MASK OVERLAY ───────────────────────────────────
        frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

        # ─── FRUIT COUNT PANEL (top left) ─────────────────────────
        panel_h = 35 + (len(fruit_counter) + 1) * 28 + 10
        cv2.rectangle(frame, (0, 0), (200, panel_h), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, 0), (200, panel_h), (255, 255, 255), 1)

        cv2.putText(frame, "Fruit Count", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        panel_y = 53
        for fruit, count in sorted(fruit_counter.items()):
            color = [c for k, c in colors.items() if names[k] == fruit][0]
            cv2.putText(frame, f"{fruit}: {count}", (10, panel_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            panel_y += 28

        total = sum(fruit_counter.values())
        cv2.putText(frame, f"Total: {total}", (10, panel_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # ─── FPS COUNTER (top right) ──────────────────────────────
        fps_text = f"Fruits: {total}"
        cv2.putText(frame, fps_text, (frame.shape[1] - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # ─── SHOW FRAME ───────────────────────────────────────────
        cv2.imshow("🍎 Fruit Detection System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Camera closed")

if __name__ == '__main__':
    main()