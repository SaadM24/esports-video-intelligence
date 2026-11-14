
import cv2
from pathlib import Path
from ultralytics import YOLO


def detect_players():

    model = YOLO("yolo11n.pt")

    input_dir = Path("data/processed/frames")
    output_dir = Path("outputs/detections")
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(input_dir.glob("*.jpg"))

    if not images:
        raise FileNotFoundError("No extracted frames found.")

    total_detections = 0

    print(f"Processing {len(images)} frames...")

    for index, image_path in enumerate(images):

        results = model.predict(
            source=str(image_path),
            classes=[0],
            conf=0.25,
            imgsz=640,
            device=0,
            verbose=False
        )

        result = results[0]

        annotated_frame = result.plot()

        detections = len(result.boxes)
        total_detections += detections

        output_path = output_dir / image_path.name

        cv2.imwrite(
            str(output_path),
            annotated_frame
        )

        if (index + 1) % 25 == 0:
            print(
                f"Processed {index + 1}/{len(images)} frames"
            )

    print("\n--- DETECTION COMPLETE ---")
    print(f"Frames processed: {len(images)}")
    print(f"Person detections: {total_detections}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    detect_players()
