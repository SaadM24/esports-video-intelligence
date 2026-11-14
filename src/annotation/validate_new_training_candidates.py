
from pathlib import Path

IMAGE_DIR = Path("data/processed/new_training_candidates")
LABEL_DIR = Path("data/annotations/new_training_candidates")

SELECTED_FRAMES = [
    24, 57, 59, 64, 92, 93, 97, 194,
    250, 251, 252, 290, 291, 311, 312, 315
]

errors = []
total_boxes = 0
empty_labels = 0

print("\n========== ANNOTATION VALIDATION ==========")

for number in SELECTED_FRAMES:
    filename = f"frame_{number:03d}.00s"
    image_path = IMAGE_DIR / f"{filename}.jpg"
    label_path = LABEL_DIR / f"{filename}.txt"

    if not image_path.exists():
        errors.append(f"Missing image: {image_path}")
        continue

    if not label_path.exists():
        errors.append(f"Missing label: {label_path}")
        continue

    lines = label_path.read_text(encoding="utf-8").splitlines()

    if not lines:
        empty_labels += 1
        print(f"{filename}: EMPTY")
        continue

    frame_boxes = 0

    for line_number, line in enumerate(lines, start=1):
        parts = line.split()

        if len(parts) != 5:
            errors.append(
                f"{filename}, line {line_number}: "
                "Expected 5 values"
            )
            continue

        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
        except ValueError:
            errors.append(
                f"{filename}, line {line_number}: Invalid number"
            )
            continue

        if class_id != 0:
            errors.append(f"{filename}: Invalid class ID {class_id}")

        if not all(0 <= value <= 1 for value in
                   [x_center, y_center, width, height]):
            errors.append(f"{filename}: Coordinate outside 0–1")

        if width <= 0 or height <= 0:
            errors.append(f"{filename}: Invalid box dimensions")

        if (x_center - width / 2 < -1e-6 or
            x_center + width / 2 > 1.000001 or
            y_center - height / 2 < -1e-6 or
            y_center + height / 2 > 1.000001):
            errors.append(f"{filename}: Bounding box exceeds image")

        frame_boxes += 1

    total_boxes += frame_boxes
    print(f"{filename}: {frame_boxes} player boxes")

print("\n========== SUMMARY ==========")
print(f"Expected frames: {len(SELECTED_FRAMES)}")
print(f"Empty labels: {empty_labels}")
print(f"Player boxes: {total_boxes}")
print(f"Errors: {len(errors)}")

for error in errors:
    print("ERROR:", error)

if not errors and empty_labels == 1:
    print("\nRESULT: ALL AUTOMATED CHECKS PASSED")
else:
    print("\nRESULT: PLEASE REVIEW ERRORS")
