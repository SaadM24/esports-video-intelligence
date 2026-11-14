
from pathlib import Path
from ultralytics import YOLO

# Load V4 model
MODEL_PATH = Path(
    "runs/detect/runs/player_detection/yolo11n_expanded_v4/weights/best.pt"
)

# Validation images
SOURCE_PATH = Path(
    "data/processed/yolo_dataset/images/val"
)

# Check required files
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not SOURCE_PATH.exists():
    raise FileNotFoundError(f"Validation images not found: {SOURCE_PATH}")

# Load model
model = YOLO(str(MODEL_PATH))

# Run predictions
results = model.predict(
    source=str(SOURCE_PATH),
    conf=0.30,
    iou=0.50,
    imgsz=640,
    device=0,
    save=True,
    project="outputs",
    name="expanded_v4_tuned",
    exist_ok=False,
    verbose=True
)

# Inspect detections on selected frames
inspect_frames = {
    "frame_228.00s.jpg",
    "frame_287.00s.jpg",
    "frame_407.00s.jpg"
}

print("\n--- DETECTION DIAGNOSTICS ---")

for result in results:
    filename = Path(result.path).name

    if filename in inspect_frames:
        print(f"\n{filename}")

        for i, box in enumerate(result.boxes):
            coordinates = box.xyxy[0].cpu().tolist()
            confidence = float(box.conf[0])

            print(
                f"Detection {i + 1}: "
                f"Confidence={confidence:.3f}, "
                f"Coordinates={coordinates}"
            )

# Inference summary
total_detections = sum(len(result.boxes) for result in results)

print("\n--- INFERENCE SUMMARY ---")
print(f"Model: {MODEL_PATH}")
print(f"Validation images: {len(results)}")
print(f"Total detections: {total_detections}")
print(f"Predictions saved to: {results[0].save_dir if results else 'No results'}")
