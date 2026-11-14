
from pathlib import Path
import pandas as pd

RUNS = {
    "V1": Path("outputs/tracking/v1_test"),
    "V5": Path("outputs/tracking/v5_test"),
}

summary = []
all_tracks = []

for model, folder in RUNS.items():
    frames = pd.read_csv(folder / "frame_summary.csv")
    tracks = pd.read_csv(folder / "player_tracks.csv")

    unique_ids = tracks["track_id"].nunique()
    frames_with_tracks = int((frames["tracked_boxes"] > 0).sum())
    unassigned = int(
        (frames["detections"] - frames["tracked_boxes"])
        .clip(lower=0).sum()
    )

    summary.append({
        "Model": model,
        "Frames": len(frames),
        "Detections": int(frames["detections"].sum()),
        "TrackedDetections": int(frames["tracked_boxes"].sum()),
        "UnassignedDetections": unassigned,
        "FramesWithTracks": frames_with_tracks,
        "DistinctTrackIDs": unique_ids,
    })

    for track_id, group in tracks.groupby("track_id"):
        frame_numbers = sorted(
            group["frame_number"].astype(int).unique()
        )

        longest = 0
        current = 0
        previous = None

        for frame in frame_numbers:
            if previous is not None and frame == previous + 1:
                current += 1
            else:
                current = 1

            longest = max(longest, current)
            previous = frame

        all_tracks.append({
            "Model": model,
            "TrackID": track_id,
            "FirstFrame": frame_numbers[0],
            "LastFrame": frame_numbers[-1],
            "ObservedFrames": len(frame_numbers),
            "LongestContinuousFrames": longest,
            "LongestContinuousSeconds": round(longest / 30, 3),
            "AverageConfidence": round(
                group["confidence"].mean(), 3
            ),
        })

summary_df = pd.DataFrame(summary)
tracks_df = pd.DataFrame(all_tracks)

print("\n========== TRACKING COMPARISON ==========")
print(summary_df.to_string(index=False))

print("\n========== PER-TRACK ANALYSIS ==========")
print(
    tracks_df.sort_values(
        ["Model", "ObservedFrames"],
        ascending=[True, False]
    ).to_string(index=False)
)

output = Path("outputs/tracking")
summary_df.to_csv(output / "tracking_comparison.csv", index=False)
tracks_df.to_csv(output / "track_continuity.csv", index=False)

print("\nAnalysis completed.")
