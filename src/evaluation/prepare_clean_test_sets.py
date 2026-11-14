
from pathlib import Path
import shutil
import yaml

SOURCE = Path("data/processed/heldout_eval")
OUTPUT_ROOT = Path("data/processed/clean_test_sets")

TEST_FRAMES = [
    6, 18, 22, 34, 40, 54, 60, 65, 70, 75,
    100, 105, 114, 132, 134, 161, 172, 198,
    203, 210, 214, 219, 225, 227, 275, 304,
    310, 323, 330, 338, 344, 352, 359, 361, 387
]

EXACT_OVERLAPS = {22, 198, 344}

BUFFERED_FRAMES = {
    105, 161, 172, 275, 323, 330,
    338, 352, 359, 361, 387
}

DATASETS = {
    "exact_overlap_free": [
        n for n in TEST_FRAMES
        if n not in EXACT_OVERLAPS
    ],
    "five_second_buffered": [
        n for n in TEST_FRAMES
        if n in BUFFERED_FRAMES
    ],
}

# Verify all inputs before copying anything.
for dataset_name, frames in DATASETS.items():
    for number in frames:
        base = f"frame_{number:03d}.00s"

        image = SOURCE / "images/test" / f"{base}.jpg"
        label = SOURCE / "labels/test" / f"{base}.txt"

        if not image.is_file() or not label.is_file():
            raise FileNotFoundError(
                f"{dataset_name}: missing image or label for {base}"
            )

for dataset_name, frames in DATASETS.items():
    destination = OUTPUT_ROOT / dataset_name

    # Stop if outputs already exist, rather than overwrite them.
    if destination.exists():
        raise FileExistsError(
            f"Output already exists: {destination}"
        )

for dataset_name, frames in DATASETS.items():

    destination = OUTPUT_ROOT / dataset_name
    images_dir = destination / "images/test"
    labels_dir = destination / "labels/test"

    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    boxes = 0
    empty_labels = 0

    for number in frames:
        base = f"frame_{number:03d}.00s"

        image_name = f"{base}.jpg"
        label_name = f"{base}.txt"

        shutil.copy2(
            SOURCE / "images/test" / image_name,
            images_dir / image_name
        )

        shutil.copy2(
            SOURCE / "labels/test" / label_name,
            labels_dir / label_name
        )

        lines = [
            line for line in
            (labels_dir / label_name).read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]

        boxes += len(lines)
        if not lines:
            empty_labels += 1

    config = {
        "path": str(destination.resolve()),
        "train": "images/test",
        "val": "images/test",
        "test": "images/test",
        "names": {0: "player"}
    }

    with (destination / "data.yaml").open(
        "w", encoding="utf-8"
    ) as file:
        yaml.safe_dump(config, file, sort_keys=False)

    print(f"\nDataset: {dataset_name}")
    print(f"Images: {len(frames)}")
    print(f"Player boxes: {boxes}")
    print(f"Empty labels: {empty_labels}")
    print(f"Saved to: {destination}")

print("\nBoth test subsets prepared successfully.")
