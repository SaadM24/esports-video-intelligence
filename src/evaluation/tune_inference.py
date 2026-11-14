from pathlib import Path
from ultralytics import YOLO

# V1 trained model
MODEL_PATH = Path(
    "runs/detect/runs/player_detection/yolo11n_expanded/weights/best.pt"
)

# Existing validation images
VAL_IMAGES = Path("data/processed/yolo_dataset/images/val")

# Output directory
OUTPUT_DIR = "runs/inference_tuning"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not VAL_IMAGES.exists():
    raise FileNotFoundError(f"Validation images not found: {VAL_IMAGES}")

# Load V1 once
model = YOLO(str(MODEL_PATH))

# Inference configurations
experiments = [
    {"name": "A_conf025_iou030", "conf": 0.25, "iou": 0.30},
    {"name": "B_conf030_iou030", "conf": 0.30, "iou": 0.30},
    {"name": "C_conf035_iou030", "conf": 0.35, "iou": 0.30},
    {
    "name": "D_conf025_iou010",
    "conf": 0.25,
    "iou": 0.10
},
{
    "name": "E_conf027_iou010",
    "conf": 0.27,
    "iou": 0.10
},
]

for experiment in experiments:
    print("\n" + "=" * 55)
    print(f"Experiment: {experiment['name']}")
    print(f"Confidence: {experiment['conf']}")
    print(f"IoU: {experiment['iou']}")
    print("=" * 55)

    results = model.predict(
        source=str(VAL_IMAGES),
        conf=experiment["conf"],
        iou=experiment["iou"],
        imgsz=640,
        device=0,
        save=True,
        save_txt=True,
        save_conf=True,
        project=OUTPUT_DIR,
        name=experiment["name"],
        exist_ok=True,
        verbose=False
    )

    total_detections = 0

    for result in results:
        count = len(result.boxes)
        total_detections += count

        print(f"{Path(result.path).name}: {count} detections")

    print(f"\nTotal detections: {total_detections}")

print("\nAll inference experiments completed!")