import cv2
from ultralytics import YOLO

# Test all three models
models_to_test = {
    "roboflow_ball": "models/roboflow_ball.pt",
    "yolov8s": "yolov8s.pt",
    "roboflow_players": "models/roboflow_players.pt"
}

cap = cv2.VideoCapture("/tmp/test_5s.mp4")

print("="*80)
print("BALL DETECTION MODEL COMPARISON")
print("="*80)
print()

for model_name, model_path in models_to_test.items():
    print(f"\n{model_name} ({model_path})")
    print("-" * 60)

    model = YOLO(model_path)
    print(f"Classes: {model.names}")

    frame_count = 0
    detection_count = 0
    class_counts = {}
    sample_detections = []

    cap = cv2.VideoCapture("/tmp/test_5s.mp4")  # Re-open for each model

    while cap.isOpened() and frame_count < 100:
        ret, frame = cap.read()
        if not ret:
            break

        # Run with low confidence — raw model output
        results = model(frame, conf=0.05, verbose=False)[0]

        if len(results.boxes) > 0:
            detection_count += 1
            for box in results.boxes:
                cls_id = int(box.cls.item())
                class_name = model.names[cls_id]
                conf = float(box.conf.item())
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

                # Store first 3 detections for display
                if len(sample_detections) < 3:
                    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                    sample_detections.append({
                        "frame": frame_count,
                        "class": class_name,
                        "conf": conf,
                        "size": f"{int(x2-x1)}x{int(y2-y1)}px"
                    })

        frame_count += 1

    print(f"Frames scanned: {frame_count}")
    print(f"Frames with detections: {detection_count} ({100*detection_count/max(frame_count,1):.1f}%)")
    print(f"Class distribution: {class_counts}")

    if sample_detections:
        print(f"\nSample detections:")
        for det in sample_detections:
            print(f"  Frame {det['frame']}: class={det['class']}, conf={det['conf']:.3f}, {det['size']}")
    else:
        print(f"NO DETECTIONS FOUND")

    del model

cap.release()

print("\n" + "="*80)
print("SUMMARY:")
print("Which model detects the ball best?")
print("="*80)
