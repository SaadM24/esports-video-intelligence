
from pathlib import Path
from ultralytics import YOLO
import torch
import pandas as pd

DATASET_PATH = Path("data/processed/yolo_dataset/data.yaml")

MODELS = {
    "V1": Path(
        "runs/detect/runs/player_detection/"
        "yolo11n_expanded/weights/best.pt"
    ),
    "V5": Path(
        "runs/detect/runs/player_detection/"
        "yolo11n_expanded_v5/weights/best.pt"
    ),
}

OUTPUT_PATH = Path("outputs/v1_v5_validation_comparison.csv")

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset configuration not found: {DATASET_PATH}"
    )

for name, model_path in MODELS.items():
    if not model_path.exists():
        raise FileNotFoundError(
            f"{name} checkpoint not found: {model_path}"
        )

device = 0 if torch.cuda.is_available() else "cpu"

print(f"\nEvaluation device: {device}")

results = []

for name, model_path in MODELS.items():

    print(f"\n{'=' * 55}")
    print(f"Evaluating {name}")
    print(f"Checkpoint: {model_path}")
    print(f"{'=' * 55}")

    model = YOLO(str(model_path))

    metrics = model.val(
        data=str(DATASET_PATH),
        split="val",
        imgsz=640,
        batch=4,
        device=device,
        workers=0,
        plots=True,
        project="runs/evaluation",
        name=f"{name}_validation_comparison",
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

print("\n==========================================")
print("        V1 vs V5 VALIDATION RESULTS")
print("==========================================")

print(
    df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print(f"\nResults saved to: {OUTPUT_PATH}")
