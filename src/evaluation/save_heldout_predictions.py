
from pathlib import Path
from ultralytics import YOLO
import torch

MODEL_PATH = Path(
    "runs/detect/runs/player_detection/"
    "yolo11n_expanded/weights/best.pt"
)

IMAGE_DIR = Path(
    "data/processed/heldout_eval/images/test"
)

model = YOLO(str(MODEL_PATH))

device = 0 if torch.cuda.is_available() else "cpu"

results = model.predict(
    source=str(IMAGE_DIR),
    imgsz=640,
    conf=0.27,
    iou=0.10,
    device=device,
    save=True,
    save_txt=True,
    save_conf=True,
    project="runs/heldout_predictions",
    name="V1_conf027_iou010",
    exist_ok=True,
    verbose=False
)

print("\nPrediction export completed.")
print(f"Images processed: {len(results)}")
print(
    "Prediction labels: "
    "runs/heldout_predictions/V1_conf027_iou010/labels"
)
