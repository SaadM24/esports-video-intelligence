
import cv2
from pathlib import Path
import math

IMAGE_DIR = Path("data/processed/frames")
OUTPUT_DIR = Path("outputs/contact_sheets")

THUMB_WIDTH = 360
THUMB_HEIGHT = 203
COLUMNS = 4
ROWS = 5
FRAMES_PER_SHEET = COLUMNS * ROWS

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

image_paths = sorted(IMAGE_DIR.glob("*.jpg"))

if not image_paths:
    raise FileNotFoundError("No extracted frames found.")

total_sheets = math.ceil(len(image_paths) / FRAMES_PER_SHEET)

print(f"Total frames: {len(image_paths)}")
print(f"Contact sheets to create: {total_sheets}")

for sheet_index in range(total_sheets):

    start = sheet_index * FRAMES_PER_SHEET
    end = start + FRAMES_PER_SHEET

    batch = image_paths[start:end]

    sheet = 255 * (
        __import__("numpy").ones(
            (ROWS * (THUMB_HEIGHT + 30),
             COLUMNS * THUMB_WIDTH, 3),
            dtype="uint8"
        )
    )

    for index, image_path in enumerate(batch):

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        thumbnail = cv2.resize(
            image,
            (THUMB_WIDTH, THUMB_HEIGHT)
        )

        row = index // COLUMNS
        col = index % COLUMNS

        x = col * THUMB_WIDTH
        y = row * (THUMB_HEIGHT + 30)

        sheet[y:y + THUMB_HEIGHT, x:x + THUMB_WIDTH] = thumbnail

        cv2.putText(
            sheet,
            image_path.stem,
            (x + 5, y + THUMB_HEIGHT + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1
        )

    output_path = OUTPUT_DIR / f"contact_sheet_{sheet_index + 1:02d}.jpg"

    cv2.imwrite(str(output_path), sheet)

    print(f"Saved: {output_path}")

print("\nContact sheet generation complete.")
