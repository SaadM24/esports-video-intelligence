
import cv2
import numpy as np
from pathlib import Path
import math

IMAGE_DIR = Path("data/processed/new_training_candidates")
LABEL_DIR = Path("data/annotations/new_training_candidates")
OUTPUT_DIR = Path("outputs/annotation_review/new_training_batch")

SELECTED_FRAMES = [
    24, 57, 59, 64, 92, 93, 97, 194,
    250, 251, 252, 290, 291, 311, 312, 315
]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

review_images = []
total_boxes = 0
empty_frames = 0
errors = []

print("\n====================================")
print("     NEW ANNOTATION VISUAL REVIEW")
print("====================================")

for frame_number in SELECTED_FRAMES:

    frame_name = f"frame_{frame_number:03d}.00s.jpg"
    label_name = f"frame_{frame_number:03d}.00s.txt"

    image_path = IMAGE_DIR / frame_name
    label_path = LABEL_DIR / label_name

    if not image_path.exists():
        errors.append(f"Missing image: {frame_name}")
        continue

    if not label_path.exists():
        errors.append(f"Missing label: {label_name}")
        continue

    image = cv2.imread(str(image_path))

    if image is None:
        errors.append(f"Cannot read image: {frame_name}")
        continue

    height, width = image.shape[:2]

    label_lines = label_path.read_text(
        encoding="utf-8"
    ).strip().splitlines()

    box_count = 0

    for line in label_lines:

        if not line.strip():
            continue

        values = line.split()

        if len(values) != 5:
            errors.append(f"Invalid label format: {label_name}")
            continue

        class_id, xc, yc, bw, bh = map(float, values)

        x1 = int((xc - bw / 2) * width)
        y1 = int((yc - bh / 2) * height)
        x2 = int((xc + bw / 2) * width)
        y2 = int((yc + bh / 2) * height)

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

        cv2.putText(
            image,
            "PLAYER",
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        box_count += 1

    total_boxes += box_count

    if box_count == 0:
        empty_frames += 1

    print(f"Frame {frame_number:03d}: {box_count} boxes")

    # Resize all images to consistent dimensions
    scale = min(480 / width, 270 / height)

    display = cv2.resize(
        image,
        (int(width * scale), int(height * scale))
    )

    cv2.putText(
        display,
        f"Frame {frame_number:03d} | Boxes: {box_count}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )

    review_images.append(display)

    cv2.imwrite(
        str(OUTPUT_DIR / frame_name),
        image
    )

# Generate 4-column contact sheet
if review_images:

    columns = 4
    rows = math.ceil(len(review_images) / columns)

    cell_height, cell_width = review_images[0].shape[:2]

    contact_sheet = np.full(
        (rows * cell_height, columns * cell_width, 3),
        255,
        dtype=np.uint8
    )

    for index, img in enumerate(review_images):

        row = index // columns
        col = index % columns

        y = row * cell_height
        x = col * cell_width

        contact_sheet[
            y:y + cell_height,
            x:x + cell_width
        ] = img

    contact_path = OUTPUT_DIR / "new_training_contact_sheet.jpg"

    cv2.imwrite(str(contact_path), contact_sheet)

    print("\nContact sheet saved:")
    print(contact_path)

print("\n====================================")
print("          REVIEW SUMMARY")
print("====================================")
print(f"Frames reviewed: {len(review_images)}")
print(f"Total bounding boxes: {total_boxes}")
print(f"Empty frames: {empty_frames}")
print(f"Errors: {len(errors)}")

for error in errors:
    print("ERROR:", error)

print(f"\nIndividual images: {OUTPUT_DIR}")
print("====================================")
