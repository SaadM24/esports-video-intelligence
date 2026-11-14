
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = Path(
    "runs/detect/runs/player_detection/"
    "yolo11n_expanded/weights/best.pt"
)

SOURCE_PATH = Path("data/processed/yolo_dataset/images/val")

model = YOLO(str(MODEL_PATH))

results = model.predict(
    source=str(SOURCE_PATH),
    imgsz=640,
    conf=0.25,
    iou=0.5,
    save=True,
    project="runs/evaluation",
    name="V1_visual_review",
    exist_ok=True,
    device=0,
    verbose=True
)

print("\nVisual review completed.")
print("Review images in: runs/evaluation/V1_visual_review")
print("Images processed:", len(results))
