
import cv2
from pathlib import Path

IMAGE_DIR = Path("data/processed/frames")
LABEL_DIR = Path("data/annotations/labels")
OUTPUT_DIR = Path("outputs/annotation_review")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

label_files = sorted(LABEL_DIR.glob("*.txt"))

total_images = 0
total_boxes = 0

for label_path in label_files:

    image_path = IMAGE_DIR / f"{label_path.stem}.jpg"

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Image missing: {image_path.name}")
        continue

    height, width = image.shape[:2]

    lines = label_path.read_text(
        encoding="utf-8"
    ).splitlines()

    box_count = 0

    for line in lines:

        if not line.strip():
            continue

        values = list(map(float, line.split()))

        class_id, xc, yc, bw, bh = values

        x1 = int((xc - bw / 2) * width)
        y1 = int((yc - bh / 2) * height)
        x2 = int((xc + bw / 2) * width)
        y2 = int((yc + bh / 2) * height)

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            image,
            "player",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        box_count += 1

    output_path = OUTPUT_DIR / image_path.name
    cv2.imwrite(str(output_path), image)

    total_images += 1
    total_boxes += box_count

    print(f"{image_path.name}: {box_count} boxes")

print("\n--- ANNOTATION REVIEW COMPLETE ---")
print(f"Images reviewed: {total_images}")
print(f"Total boxes: {total_boxes}")
print(f"Saved to: {OUTPUT_DIR}")
