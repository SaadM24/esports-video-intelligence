
import cv2
from pathlib import Path


def extract_frames(video_path, interval_seconds=1, output_dir="data/processed/frames"):

    video_path = Path(video_path)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError("Could not open video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_interval = max(1, round(fps * interval_seconds))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_number = 0
    saved_count = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        if frame_number % frame_interval == 0:

            timestamp = frame_number / fps

            filename = (
                f"frame_{timestamp:06.2f}s.jpg"
            )

            output_path = output_dir / filename

            cv2.imwrite(str(output_path), frame)

            saved_count += 1

        frame_number += 1

    cap.release()

    print("\n--- FRAME EXTRACTION COMPLETE ---")
    print(f"Original frames: {total_frames}")
    print(f"Extracted frames: {saved_count}")
    print(f"Sampling interval: {interval_seconds} second(s)")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":

    extract_frames(
        "data/raw/gameplay.mp4",
        interval_seconds=1
    )
