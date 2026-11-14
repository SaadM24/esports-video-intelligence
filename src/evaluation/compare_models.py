
from pathlib import Path
from ultralytics import YOLO
import torch
import pandas as pd

DATASET_PATH = Path("data/processed/yolo_dataset/data.yaml")

MODELS = {
    "V1": Path("runs/detect/runs/player_detection/yolo11n_expanded/weights/best.pt"),
    "V2": Path("runs/detect/runs/player_detection/yolo11n_expanded_v2/weights/best.pt"),
    "V3": Path("runs/detect/runs/player_detection/yolo11n_expanded_v3/weights/best.pt"),
    "V4": Path("runs/detect/runs/player_detection/yolo11n_expanded_v4/weights/best.pt"),
}

if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

device = 0 if torch.cuda.is_available() else "cpu"

print(f"Evaluation device: {device}")
print("\nChecking model checkpoints...")

for name, path in MODELS.items():
    if not path.exists():
        raise FileNotFoundError(f"{name} checkpoint not found: {path}")
    print(f"{name}: Found")

results = []

for name, path in MODELS.items():
    print(f"\n{'=' * 50}")
    print(f"Evaluating {name}")
    print(f"Checkpoint: {path}")
    print(f"{'=' * 50}")

    model = YOLO(str(path))

    metrics = model.val(
        data=str(DATASET_PATH),
        split="val",
        imgsz=640,
        batch=4,
        device=device,
        workers=0,
        plots=True,
        project="runs/evaluation",
        name=name,
        exist_ok=True,
        verbose=True
    )

    results.append({
        "Model": name,
        "Precision": metrics.box.mp,
        "Recall": metrics.box.mr,
        "mAP50": metrics.box.map50,
        "mAP50-95": metrics.box.map,
    })

df = pd.DataFrame(results)

print("\n\nFINAL MODEL COMPARISON")
print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

output_path = Path("outputs/model_comparison.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

print(f"\nComparison saved to: {output_path}")
