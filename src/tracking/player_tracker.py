
import csv
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

# ==========================================
# CONFIGURATION
# ==========================================

VIDEO_PATH = Path("data/raw/gameplay.mp4")

MODEL_PATH = Path(
    "runs/detect/runs/player_detection/"
    "yolo11n_expanded_v5/weights/best.pt"
)

OUTPUT_DIR = Path("outputs/tracking/v5_test")

START_SECOND = 248
DURATION_SECONDS = 10

CONFIDENCE = 0.20
NMS_IOU = 0.50
IMAGE_SIZE = 640

OUTPUT_VIDEO = OUTPUT_DIR / "tracked_gameplay.mp4"
OUTPUT_CSV = OUTPUT_DIR / "player_tracks.csv"
FRAME_CSV = OUTPUT_DIR / "frame_summary.csv"

# ==========================================
# VALIDATION
# ==========================================

if not VIDEO_PATH.is_file():
    raise FileNotFoundError(f"Video missing: {VIDEO_PATH}")

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Model missing: {MODEL_PATH}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for path in (OUTPUT_VIDEO, OUTPUT_CSV, FRAME_CSV):
    if path.exists():
        raise FileExistsError(
            f"Output already exists: {path}. "
            "Use a new output folder for another run."
        )

device = 0 if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")
print(f"Model: {MODEL_PATH}")

model = YOLO(str(MODEL_PATH))

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError("Unable to open gameplay video.")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if fps <= 0 or width <= 0 or height <= 0:
    cap.release()
    raise RuntimeError("Invalid video properties.")

start_frame = int(START_SECOND * fps)
end_frame = min(
    total_frames,
    start_frame + int(DURATION_SECONDS * fps)
)

if start_frame >= total_frames:
    cap.release()
    raise ValueError("Start time is beyond video duration.")

cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError("Unable to create output video.")

print("\n========== VIDEO INFORMATION ==========")
print(f"Resolution: {width} x {height}")
print(f"FPS: {fps:.2f}")
print(f"Video duration: {total_frames / fps:.2f}s")
print(f"Clip: {START_SECOND}s to {end_frame / fps:.2f}s")
print("Tracker: ByteTrack")
print("=======================================\n")

processed_frames = 0
total_detections = 0
total_tracked_boxes = 0
unique_track_ids = set()

try:
    with (
        OUTPUT_CSV.open("w", newline="", encoding="utf-8") as tracks_file,
        FRAME_CSV.open("w", newline="", encoding="utf-8") as summary_file
    ):

        tracks_writer = csv.writer(tracks_file)
        summary_writer = csv.writer(summary_file)

        tracks_writer.writerow([
            "frame_number",
            "timestamp_seconds",
            "track_id",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2"
        ])

        summary_writer.writerow([
            "frame_number",
            "timestamp_seconds",
            "detections",
            "tracked_boxes",
            "unique_ids_so_far"
        ])

        for frame_number in range(start_frame, end_frame):

            success, frame = cap.read()

            if not success:
                print("Video ended earlier than expected.")
                break

            result = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=CONFIDENCE,
                iou=NMS_IOU,
                imgsz=IMAGE_SIZE,
                device=device,
                verbose=False
            )[0]

            timestamp = frame_number / fps
            boxes = result.boxes

            detections = len(boxes) if boxes is not None else 0

            has_ids = (
                boxes is not None
                and boxes.is_track
                and boxes.id is not None
            )

            tracked_count = 0

            if has_ids:
                coordinates = boxes.xyxy.cpu().numpy()
                confidences = boxes.conf.cpu().numpy()
                track_ids = boxes.id.int().cpu().tolist()

                for box, confidence, track_id in zip(
                    coordinates, confidences, track_ids
                ):
                    x1, y1, x2, y2 = box.tolist()

                    tracks_writer.writerow([
                        frame_number,
                        round(timestamp, 3),
                        track_id,
                        round(float(confidence), 4),
                        round(x1, 1),
                        round(y1, 1),
                        round(x2, 1),
                        round(y2, 1)
                    ])

                    unique_track_ids.add(track_id)
                    tracked_count += 1

            total_detections += detections
            total_tracked_boxes += tracked_count

            summary_writer.writerow([
                frame_number,
                round(timestamp, 3),
                detections,
                tracked_count,
                len(unique_track_ids)
            ])

            annotated_frame = result.plot()

            cv2.putText(
                annotated_frame,
                f"Time: {timestamp:.2f}s",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2
            )

            cv2.putText(
                annotated_frame,
                f"Tracked players: {tracked_count}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2
            )

            writer.write(annotated_frame)
            processed_frames += 1

            if processed_frames % 30 == 0:
                print(
                    f"Processed {processed_frames} frames | "
                    f"Current tracks: {tracked_count} | "
                    f"Unique IDs: {len(unique_track_ids)}"
                )

finally:
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

print("\n========== TRACKING SUMMARY ==========")
print(f"Frames processed: {processed_frames}")
print(f"Total detections across frames: {total_detections}")
print(f"Detections with tracking IDs: {total_tracked_boxes}")
print(f"Distinct track IDs assigned: {len(unique_track_ids)}")
print(f"Output video: {OUTPUT_VIDEO}")
print(f"Track CSV: {OUTPUT_CSV}")
print(f"Frame CSV: {FRAME_CSV}")
print("======================================")
