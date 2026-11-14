
import cv2
from pathlib import Path

# --------------------------------------------------
# DIRECTORIES
# --------------------------------------------------

IMAGE_DIR = Path("data/processed/frames")
LABEL_DIR = Path("data/annotations/labels")

# --------------------------------------------------
# NEW FRAMES TO ANNOTATE
# The original 29 annotated frames are not included.
# --------------------------------------------------

SELECTED_FRAMES = [20, 21, 22, 28, 29, 30, 35, 37]

# Maximum display dimensions
MAX_DISPLAY_WIDTH = 1200
MAX_DISPLAY_HEIGHT = 750

LABEL_DIR.mkdir(parents=True, exist_ok=True)

total_boxes = 0
completed_frames = 0
skipped_frames = 0

print("\n====================================")
print("     CS2 PLAYER ANNOTATION")
print("====================================")
print("\nAnnotation instructions:")
print("- Draw a box around every visible living player.")
print("- Include teammates and opponents.")
print("- Exclude dead bodies.")
print("- Exclude first-person hands and weapons.")
print("- Exclude HUD elements and shadows.")
print("\nControls:")
print("- ENTER or SPACE: confirm a box")
print("- C or an empty selection: finish the frame")
print("====================================\n")


# --------------------------------------------------
# ANNOTATION LOOP
# --------------------------------------------------

for frame_number in SELECTED_FRAMES:

    frame_name = f"frame_{frame_number:03d}.00s.jpg"

    image_path = IMAGE_DIR / frame_name
    label_path = LABEL_DIR / f"frame_{frame_number:03d}.00s.txt"

    # Check image exists
    if not image_path.exists():
        print(f"[MISSING IMAGE] {frame_name}")
        continue

    # Protect existing annotations
    if label_path.exists():
        print(f"[SKIPPED - LABEL ALREADY EXISTS] {label_path.name}")
        skipped_frames += 1
        continue

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"[ERROR] Could not read {frame_name}")
        continue

    height, width = image.shape[:2]

    # Resize for display only
    scale = min(
        MAX_DISPLAY_WIDTH / width,
        MAX_DISPLAY_HEIGHT / height,
        1.0
    )

    display_width = int(width * scale)
    display_height = int(height * scale)

    display_image = cv2.resize(
        image,
        (display_width, display_height)
    )

    boxes = []

    print(f"\nAnnotating: {frame_name}")
    print("Select each living player one at a time.")
    print("Finish the frame with C or an empty selection.")

    while True:

        roi = cv2.selectROI(
            "CS2 Player Annotation",
            display_image,
            showCrosshair=True,
            fromCenter=False
        )

        x, y, box_width, box_height = roi

        # Empty selection means finish this frame
        if box_width == 0 or box_height == 0:
            break

        # Convert display coordinates to original coordinates
        x1 = x / scale
        y1 = y / scale
        x2 = (x + box_width) / scale
        y2 = (y + box_height) / scale

        # Ensure coordinates stay within image boundaries
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))

        actual_width = x2 - x1
        actual_height = y2 - y1

        if actual_width <= 0 or actual_height <= 0:
            continue

        # Convert to YOLO normalised coordinates
        x_center = ((x1 + x2) / 2) / width
        y_center = ((y1 + y2) / 2) / height
        normalized_width = actual_width / width
        normalized_height = actual_height / height

        boxes.append(
            f"0 {x_center:.6f} {y_center:.6f} "
            f"{normalized_width:.6f} {normalized_height:.6f}"
        )

        print(f"Player box added. Frame total: {len(boxes)}")

    # Save labels
    label_path.write_text(
        "\n".join(boxes),
        encoding="utf-8"
    )

    total_boxes += len(boxes)
    completed_frames += 1

    print(f"Saved: {label_path}")
    print(f"Player boxes in this frame: {len(boxes)}")

    cv2.destroyAllWindows()


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

cv2.destroyAllWindows()

print("\n====================================")
print("        ANNOTATION SUMMARY")
print("====================================")
print(f"Frames completed: {completed_frames}")
print(f"Frames skipped: {skipped_frames}")
print(f"Total new player boxes: {total_boxes}")
print(f"Labels saved in: {LABEL_DIR}")
print("====================================")
