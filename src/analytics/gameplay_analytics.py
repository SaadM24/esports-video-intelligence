
from pathlib import Path
import pandas as pd

# ==================================
# CONFIGURATION
# ==================================

INPUT_DIR = Path("outputs/tracking/v1_test")
OUTPUT_DIR = Path("outputs/analytics/v1")

FRAMES_PATH = INPUT_DIR / "frame_summary.csv"
TRACKS_PATH = INPUT_DIR / "player_tracks.csv"

FPS = 30

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================================
# LOAD TRACKING DATA
# ==================================

frames = pd.read_csv(FRAMES_PATH)
tracks = pd.read_csv(TRACKS_PATH)

if frames.empty:
    raise ValueError("Frame summary is empty.")

frames["second"] = frames["timestamp_seconds"].astype(int)

# ==================================
# PER-SECOND ANALYTICS
# ==================================

per_second = (
    frames.groupby("second")
    .agg(
        frames_processed=("frame_number", "count"),
        total_detections=("detections", "sum"),
        tracked_observations=("tracked_boxes", "sum"),
        frames_with_tracks=(
            "tracked_boxes",
            lambda x: int((x > 0).sum())
        ),
        max_simultaneous_tracks=("tracked_boxes", "max"),
    )
    .reset_index()
)

per_second["visibility_ratio"] = (
    per_second["frames_with_tracks"]
    / per_second["frames_processed"]
)

# ==================================
# TRACK-LEVEL ANALYTICS
# ==================================

track_summary = []

for track_id, group in tracks.groupby("track_id"):

    group = group.sort_values("frame_number")

    frame_numbers = sorted(
        group["frame_number"].astype(int).unique()
    )

    gaps = sum(
        current - previous > 1
        for previous, current in zip(
            frame_numbers[:-1],
            frame_numbers[1:]
        )
    )

    track_summary.append({
        "track_id": int(track_id),
        "first_seen_second": round(
            group["timestamp_seconds"].min(), 3
        ),
        "last_seen_second": round(
            group["timestamp_seconds"].max(), 3
        ),
        "observed_frames": len(frame_numbers),
        "observed_duration_seconds": round(
            len(frame_numbers) / FPS, 3
        ),
        "internal_gaps": gaps,
        "average_confidence": round(
            group["confidence"].mean(), 3
        )
    })

track_df = pd.DataFrame(track_summary)

if not track_df.empty:
    track_df = track_df.sort_values(
        "observed_frames",
        ascending=False
    )

# ==================================
# EXPORT RESULTS
# ==================================

per_second_path = OUTPUT_DIR / "per_second_analytics.csv"
track_path = OUTPUT_DIR / "track_analytics.csv"

per_second.to_csv(per_second_path, index=False)
track_df.to_csv(track_path, index=False)

print("\n========== PER-SECOND ANALYTICS ==========")
print(per_second.to_string(index=False))

print("\n========== TRACK ANALYTICS ==========")
print(track_df.to_string(index=False))

print("\nAnalytics completed.")
print(f"Per-second results: {per_second_path}")
print(f"Track results: {track_path}")
