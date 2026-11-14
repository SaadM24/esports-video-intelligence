
from pathlib import Path
import cv2

# Dataset location
DATASET_DIR = Path("data/processed/yolo_dataset")

# Floating-point tolerance for coordinate checks
TOLERANCE = 1e-6

total_images = 0
total_boxes = 0
total_empty_labels = 0
errors = []


def check_dataset():

    global total_images, total_boxes, total_empty_labels

    for split in ("train", "val"):

        image_dir = DATASET_DIR / "images" / split
        label_dir = DATASET_DIR / "labels" / split

        images = sorted(image_dir.glob("*.jpg"))
        labels = sorted(label_dir.glob("*.txt"))

        print(f"\nChecking {split.upper()} dataset...")
        print(f"Images: {len(images)}")
        print(f"Labels: {len(labels)}")

        # Check image and label counts
        if len(images) != len(labels):
            errors.append(
                f"{split}: image/label count mismatch"
            )

        # Check every image
        for image_path in images:

            total_images += 1

            label_path = label_dir / f"{image_path.stem}.txt"

            if not label_path.exists():
                errors.append(f"Missing label: {label_path}")
                continue

            # Check image readability
            image = cv2.imread(str(image_path))

            if image is None:
                errors.append(f"Unreadable image: {image_path}")
                continue

            height, width = image.shape[:2]

            # Read annotation file
            with open(label_path, "r") as file:
                lines = file.readlines()

            # Empty files are valid negative examples
            if not lines:
                total_empty_labels += 1
                continue

            # Validate each bounding box
            for line_number, line in enumerate(lines, start=1):

                values = line.strip().split()

                if len(values) != 5:
                    errors.append(
                        f"Invalid format: {label_path}, line {line_number}"
                    )
                    continue

                try:
                    class_id, x, y, w, h = map(float, values)
                except ValueError:
                    errors.append(
                        f"Non-numeric values: {label_path}, line {line_number}"
                    )
                    continue

                # Check class ID
                if class_id != 0:
                    errors.append(
                        f"Unexpected class ID: {label_path}"
                    )
                    continue

                # Check normalized coordinates
                if not all(
                    0 <= value <= 1 for value in (x, y, w, h)
                ):
                    errors.append(
                        f"Coordinates outside normalized range: {label_path}"
                    )
                    continue

                # Check bounding-box dimensions
                if w <= 0 or h <= 0:
                    errors.append(
                        f"Invalid box dimensions: {label_path}"
                    )
                    continue

                # Check image boundaries with rounding tolerance
                if (
                    x - w / 2 < -TOLERANCE
                    or x + w / 2 > 1 + TOLERANCE
                ):
                    errors.append(
                        f"Box exceeds image width: {label_path}"
                    )

                if (
                    y - h / 2 < -TOLERANCE
                    or y + h / 2 > 1 + TOLERANCE
                ):
                    errors.append(
                        f"Box exceeds image height: {label_path}"
                    )

                total_boxes += 1


def main():

    check_dataset()

    print("\n--- DATASET INTEGRITY REPORT ---")
    print(f"Images checked: {total_images}")
    print(f"Bounding boxes: {total_boxes}")
    print(f"Empty labels: {total_empty_labels}")
    print(f"Errors found: {len(errors)}")

    if errors:
        print("\nIssues found:")
        for error in errors:
            print("ERROR:", error)
    else:
        print("\nAll dataset checks passed!")


if __name__ == "__main__":
    main()
