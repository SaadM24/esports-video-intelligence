
from pathlib import Path
from ultralytics import YOLO
import torch
import pandas as pd

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

DATASETS = {
    "exact_overlap_free": Path(
        "data/processed/clean_test_sets/"
        "exact_overlap_free/data.yaml"
    ),
    "five_second_buffered": Path(
        "data/processed/clean_test_sets/"
        "five_second_buffered/data.yaml"
    ),
}

OUTPUT = Path("outputs/v1_v5_heldout_comparison.csv")

device = 0 if torch.cuda.is_available() else "cpu"

# Check all required files before evaluation
for name, path in {**MODELS, **DATASETS}.items():
    if not path.is_file():
        raise FileNotFoundError(f"{name}: {path}")

results = []

for dataset_name, dataset_path in DATASETS.items():
    for model_name, checkpoint in MODELS.items():

        experiment = f"{model_name}_{dataset_name}"

        print(f"\n{'=' * 60}")
        print(f"Evaluating: {experiment}")
        print(f"{'=' * 60}")

        model = YOLO(str(checkpoint))

        metrics = model.val(
            data=str(dataset_path),
            split="test",
            imgsz=640,
            batch=4,
            device=device,
            workers=0,
            plots=True,
            project="runs/evaluation",
            name=f"{experiment}_heldout",
            exist_ok=False,
            verbose=True,
        )

        results.append({
            "Dataset": dataset_name,
            "Model": model_name,
            "Images": 32 if dataset_name == "exact_overlap_free" else 11,
            "PlayerInstances": (
                8 if dataset_name == "exact_overlap_free" else 3
            ),
            "Precision": metrics.box.mp,
            "Recall": metrics.box.mr,
            "mAP50": metrics.box.map50,
            "mAP50-95": metrics.box.map,
        })

df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("V1 vs V5 — HELD-OUT COMPARISON")
print("=" * 70)

print(
    df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)

print(f"\nResults saved to: {OUTPUT}")
