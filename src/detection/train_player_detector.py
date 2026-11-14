
from pathlib import Path
from ultralytics import YOLO
import torch

# Project paths
MODEL_PATH = Path("yolo11n.pt")
DATASET_PATH = Path("data/processed/yolo_dataset/data.yaml")

# Check required files
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Dataset configuration not found: {DATASET_PATH}")

# Check GPU availability
if torch.cuda.is_available():
    device = 0
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    print("CUDA unavailable. Training will use CPU.")

# Load pretrained YOLO11 Nano
model = YOLO(str(MODEL_PATH))

# Train V5
results = model.train(
    data=str(DATASET_PATH),
    epochs=30,
    imgsz=640,
    batch=4,
    device=device,
    workers=0,
    patience=15,
    project="runs/player_detection",
    name="yolo11n_expanded_v5",
    exist_ok=False,
    pretrained=True,
    seed=42,
    deterministic=True,
    plots=True,
    save=True,
    verbose=True
)

print("\nTraining completed!")
print("Best model:", model.trainer.best)
