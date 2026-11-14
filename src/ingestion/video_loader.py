
import cv2
from pathlib import Path


def inspect_video(video_path: str):
    """Read video metadata and extract a sample frame."""

    path = Path(video_path)

    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        raise RuntimeError("Could not open video file.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    duration = frame_count / fps if fps > 0 else 0

    print("\n--- VIDEO INFORMATION ---")
    print(f"Filename: {path.name}")
    print(f"Resolution: {width} x {height}")
    print(f"FPS: {fps:.2f}")
    print(f"Total frames: {frame_count}")
    print(f"Duration: {duration:.2f} seconds")

    success, frame = cap.read()

    if success:
        output_dir = Path("outputs/frames")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / "first_frame.jpg"
        cv2.imwrite(str(output_path), frame)

        print(f"\nFirst frame saved to: {output_path}")
    else:
        print("Could not extract the first frame.")

    cap.release()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Inspect a gameplay video"
    )
    parser.add_argument("video", help="Path to gameplay video")

    args = parser.parse_args()
    inspect_video(args.video)
