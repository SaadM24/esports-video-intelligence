
from pathlib import Path
from ultralytics import YOLO
import torch
import pandas as pd

# Configuration
MODEL_PATH = Path(
    "runs/detect/runs/player_detection/"
    "yolo11n_expanded/weights/best.pt"
)

DATASET_PATH = Path("data/processed/heldout_eval/data.yaml")

OUTPUT_DIR = Path("outputs/heldout_evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

device = 0 if torch.cuda.is_available() else "cpu"

print("=" * 50)
print("       V1 HELD-OUT FRAME EVALUATION")
print("=" * 50)

print(f"Model: {MODEL_PATH}")
print(f"Dataset: {DATASET_PATH}")
print(f"Device: {device}")

model = YOLO(str(MODEL_PATH))

metrics = model.val(
    data=str(DATASET_PATH),
    split="test",
    imgsz=640,
    batch=4,
    device=device,
    workers=0,
    plots=True,
    project="runs/evaluation",
    name="V1_heldout",
    exist_ok=True,
    verbose=True
)

results = {
    "Model": "V1",
    "Evaluation": "Held-out frames from same recording",
    "Images": 35,
    "Ground_truth_boxes": 10,
    "Precision": metrics.box.mp,
    "Recall": metrics.box.mr,
    "mAP50": metrics.box.map50,
    "mAP50-95": metrics.box.map,
}

df = pd.DataFrame([results])

print("\n========== V1 EVALUATION RESULTS ==========")
print(df.to_string(index=False))

csv_path = OUTPUT_DIR / "v1_heldout_results.csv"
df.to_csv(csv_path, index=False)

print(f"\nResults saved to: {csv_path}")
print("Evaluation completed.")
