# class_results.py
from ultralytics import YOLO

def main():
    model = YOLO("runs/segment/fruit_seg_v33/weights/best.pt")

    metrics = model.val(
        data="fruit_final2.yaml",
        imgsz=640,
        conf=0.4,
        iou=0.5,
        verbose=True,
        workers=0      # ← fixes multiprocessing on Windows
    )

    names = {0: "Apple", 1: "Banana", 2: "Mango", 3: "Orange", 4: "Watermelon"}

    print("\n" + "="*60)
    print("       PER CLASS ACCURACY REPORT")
    print("="*60)
    print(f"\n{'Class':<12} {'Precision':>10} {'Recall':>10} {'mAP50':>10} {'mAP50-95':>10}")
    print("-"*60)

    for i, name in names.items():
        try:
            p       = metrics.box.p[i]
            r       = metrics.box.r[i]
            map50   = metrics.box.ap50[i]
            map5095 = metrics.box.ap[i]

            if map50 >= 0.90:
                grade = "🟢 Excellent"
            elif map50 >= 0.75:
                grade = "🟡 Good"
            elif map50 >= 0.60:
                grade = "🟠 Acceptable"
            else:
                grade = "🔴 Weak"

            print(f"{name:<12} {p:>10.3f} {r:>10.3f} {map50:>10.3f} {map5095:>10.3f}  {grade}")
        except Exception as e:
            print(f"{name:<12} → data not available")

    print("-"*60)
    print(f"\n{'Overall':<12} {metrics.box.mp:>10.3f} {metrics.box.mr:>10.3f} {metrics.box.map50:>10.3f} {metrics.box.map:>10.3f}")

    print("\n" + "="*60)
    print("       MASK ACCURACY REPORT")
    print("="*60)
    print(f"\n{'Class':<12} {'Precision':>10} {'Recall':>10} {'mAP50':>10} {'mAP50-95':>10}")
    print("-"*60)

    for i, name in names.items():
        try:
            p       = metrics.seg.p[i]
            r       = metrics.seg.r[i]
            map50   = metrics.seg.ap50[i]
            map5095 = metrics.seg.ap[i]

            if map50 >= 0.90:
                grade = "🟢 Excellent"
            elif map50 >= 0.75:
                grade = "🟡 Good"
            elif map50 >= 0.60:
                grade = "🟠 Acceptable"
            else:
                grade = "🔴 Weak"

            print(f"{name:<12} {p:>10.3f} {r:>10.3f} {map50:>10.3f} {map5095:>10.3f}  {grade}")
        except Exception as e:
            print(f"{name:<12} → data not available")

    print("-"*60)
    print(f"\n{'Overall':<12} {metrics.seg.mp:>10.3f} {metrics.seg.mr:>10.3f} {metrics.seg.map50:>10.3f} {metrics.seg.map:>10.3f}")
    print("\n" + "="*60)

if __name__ == '__main__':
    main()