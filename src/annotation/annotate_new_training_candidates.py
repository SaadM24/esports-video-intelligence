
import cv2
from pathlib import Path

IMAGE_DIR = Path("data/processed/new_training_candidates")
LABEL_DIR = Path("data/annotations/new_training_candidates")

SELECTED_FRAMES = [
    24, 57, 59, 64, 92, 93, 97, 194,
    250, 251, 252, 290, 291, 311, 312, 315
]

# Confirmed by visual review: no player in frame 194
EMPTY_FRAMES = {194}

MAX_DISPLAY_WIDTH = 1200
MAX_DISPLAY_HEIGHT = 750

LABEL_DIR.mkdir(parents=True, exist_ok=True)

total_boxes = 0
completed_frames = 0
skipped_frames = 0

print("\n====================================")
print("   NEW CS2 TRAINING ANNOTATIONS")
print("====================================")
print("Class 0 = player")
print("Annotate every visibly living player.")
print("Include distant players if identifiable.")
print("Exclude dead bodies, hands and X-ray outlines.")
print("ENTER/SPACE = confirm bounding box.")
print("C = finish current frame.")
print("====================================")

for frame_number in SELECTED_FRAMES:

    frame_name = f"frame_{frame_number:03d}.00s.jpg"
    image_path = IMAGE_DIR / frame_name
    label_path = LABEL_DIR / f"frame_{frame_number:03d}.00s.txt"

    if label_path.exists():
        print(f"[ALREADY ANNOTATED] {frame_name}")
        skipped_frames += 1
        continue

    if not image_path.exists():
        print(f"[MISSING IMAGE] {frame_name}")
        skipped_frames += 1
        continue

    # Create an empty label for confirmed negative frames
    if frame_number in EMPTY_FRAMES:
        label_path.write_text("", encoding="utf-8")
        completed_frames += 1
        print(f"[EMPTY LABEL SAVED] {frame_name}")
        continue

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"[ERROR] Cannot read {frame_name}")
        skipped_frames += 1
        continue

    height, width = image.shape[:2]

    scale = min(
        MAX_DISPLAY_WIDTH / width,
        MAX_DISPLAY_HEIGHT / height,
        1.0
    )

    display_image = cv2.resize(
        image,
        (int(width * scale), int(height * scale))
    )

    boxes = []

    print(f"\nAnnotating frame {frame_number:03d}")

    while True:

        roi = cv2.selectROI(
            f"Annotating Frame {frame_number:03d}",
            display_image,
            showCrosshair=True,
            fromCenter=False
        )

        x, y, box_width, box_height = roi

        if box_width == 0 or box_height == 0:
            break

        x1 = max(0, min(x / scale, width))
        y1 = max(0, min(y / scale, height))
        x2 = max(0, min((x + box_width) / scale, width))
        y2 = max(0, min((y + box_height) / scale, height))

        actual_width = x2 - x1
        actual_height = y2 - y1

        if actual_width <= 0 or actual_height <= 0:
            continue

        x_center = ((x1 + x2) / 2) / width
        y_center = ((y1 + y2) / 2) / height

        normalized_width = actual_width / width
        normalized_height = actual_height / height

        boxes.append(
            f"0 {x_center:.6f} {y_center:.6f} "
            f"{normalized_width:.6f} {normalized_height:.6f}"
        )

        print(f"Box added. Current total: {len(boxes)}")

    label_path.write_text(
        "\n".join(boxes),
        encoding="utf-8"
    )

    total_boxes += len(boxes)
    completed_frames += 1

    print(f"Saved: {label_path.name}")
    print(f"Players annotated: {len(boxes)}")

    cv2.destroyAllWindows()

cv2.destroyAllWindows()

print("\n====================================")
print("       ANNOTATION SUMMARY")
print("====================================")
print(f"Frames completed: {completed_frames}")
print(f"Frames skipped: {skipped_frames}")
print(f"Player boxes created: {total_boxes}")
print(f"Labels saved in: {LABEL_DIR}")
print("====================================")
