
import shutil
from pathlib import Path

# Existing source directories
IMAGE_DIR = Path("data/processed/frames")
LABEL_DIR = Path("data/annotations/labels")

# New dataset directory; original files remain untouched
OUTPUT_DIR = Path("data/processed/yolo_dataset")

# All 34 reviewed frames: 29 main annotations + 5 pilot frames
SELECTED_FRAMES = [
    0, 1, 2, 3, 13, 23, 31, 39, 46, 49,
    53, 56, 58, 71, 80, 90, 94, 112, 129,
    143, 192, 196, 198, 228, 229, 262, 281,
    300, 308, 313, 314, 344, 370, 407
]

# Hold out the later gameplay segment for validation
VALIDATION_FRAMES = {
    228, 229, 262, 281, 300, 308,
    313, 314, 344, 370, 407
}


def validate_label(label_path):
    """Check YOLO label formatting and count bounding boxes."""

    boxes = 0

    with open(label_path, "r") as file:
        for line_number, line in enumerate(file, start=1):
            values = line.strip().split()

            # Empty files are valid negative examples
            if not values:
                continue

            if len(values) != 5:
                raise ValueError(
                    f"Invalid format: {label_path}, line {line_number}"
                )

            class_id, x, y, width, height = map(float, values)

            if class_id != 0:
                raise ValueError(
                    f"Unexpected class ID in {label_path}"
                )

            if not all(0 <= value <= 1 for value in (x, y, width, height)):
                raise ValueError(
                    f"Coordinates outside normalized range: {label_path}"
                )

            if width <= 0 or height <= 0:
                raise ValueError(
                    f"Invalid box dimensions: {label_path}"
                )

            boxes += 1

    return boxes


def main():

    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"{OUTPUT_DIR} already exists. "
            "Check its contents before running this script again."
        )

    train_frames = []
    val_frames = []

    for frame_number in SELECTED_FRAMES:

        stem = f"frame_{frame_number:03d}.00s"

        image_path = IMAGE_DIR / f"{stem}.jpg"
        label_path = LABEL_DIR / f"{stem}.txt"

        if not image_path.exists():
            raise FileNotFoundError(f"Missing image: {image_path}")

        if not label_path.exists():
            raise FileNotFoundError(f"Missing label: {label_path}")

        if frame_number in VALIDATION_FRAMES:
            val_frames.append((image_path, label_path))
        else:
            train_frames.append((image_path, label_path))

    # Validate every label before copying anything
    total_boxes = 0

    for _, label_path in train_frames + val_frames:
        total_boxes += validate_label(label_path)

    # Create YOLO directory structure
    for split in ("train", "val"):
        (OUTPUT_DIR / "images" / split).mkdir(
            parents=True, exist_ok=True
        )
        (OUTPUT_DIR / "labels" / split).mkdir(
            parents=True, exist_ok=True
        )

    # Copy images and corresponding labels
    for split, frames in (
        ("train", train_frames),
        ("val", val_frames)
    ):
        for image_path, label_path in frames:
            shutil.copy2(
                image_path,
                OUTPUT_DIR / "images" / split / image_path.name
            )

            shutil.copy2(
                label_path,
                OUTPUT_DIR / "labels" / split / label_path.name
            )

    print("\n--- YOLO DATASET PREPARATION COMPLETE ---")
    print(f"Training images: {len(train_frames)}")
    print(f"Validation images: {len(val_frames)}")
    print(f"Total images: {len(train_frames) + len(val_frames)}")
    print(f"Total bounding boxes: {total_boxes}")
    print(f"Dataset location: {OUTPUT_DIR}")

    print("\nEmpty label files (negative examples):")
    for split, frames in (
        ("train", train_frames),
        ("val", val_frames)
    ):
        empty_count = sum(
            1 for _, label_path in frames
            if label_path.stat().st_size == 0
        )
        print(f"{split}: {empty_count}")


if __name__ == "__main__":
    main()
