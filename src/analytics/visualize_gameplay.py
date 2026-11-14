
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

INPUT_DIR = Path("outputs/analytics/v1")
OUTPUT_DIR = Path("outputs/analytics/v1/charts")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

seconds = pd.read_csv(
    INPUT_DIR / "per_second_analytics.csv"
)

tracks = pd.read_csv(
    INPUT_DIR / "track_analytics.csv"
)

# ==========================================
# 1. PLAYER VISIBILITY TIMELINE
# ==========================================

fig, ax = plt.subplots(figsize=(10, 5))

ax.bar(
    seconds["second"],
    seconds["visibility_ratio"] * 100
)

ax.set_title("Tracked Player Visibility by Second")
ax.set_xlabel("Video timestamp (seconds)")
ax.set_ylabel("Frames with tracks (%)")
ax.set_ylim(0, 105)
ax.set_xticks(seconds["second"].tolist())
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "player_visibility.png",
    dpi=160
)
plt.close(fig)

# ==========================================
# 2. DETECTION ACTIVITY
# ==========================================

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    seconds["second"],
    seconds["total_detections"],
    marker="o",
    label="All detections"
)

ax.plot(
    seconds["second"],
    seconds["tracked_observations"],
    marker="s",
    label="Detections with IDs"
)

ax.set_title("Detection Activity by Second")
ax.set_xlabel("Video timestamp (seconds)")
ax.set_ylabel("Bounding-box observations")
ax.set_xticks(seconds["second"].tolist())
ax.legend()
ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "detection_activity.png",
    dpi=160
)
plt.close(fig)

# ==========================================
# 3. TRACK OBSERVATION DURATION
# ==========================================

fig, ax = plt.subplots(figsize=(9, 5))

ax.bar(
    tracks["track_id"].astype(str),
    tracks["observed_duration_seconds"]
)

ax.set_title("Observed Duration by Tracking ID")
ax.set_xlabel("Tracking ID")
ax.set_ylabel("Observed duration (seconds)")
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "track_durations.png",
    dpi=160
)
plt.close(fig)

print("\n========== ANALYTICS VISUALIZATION ==========")
print("Generated charts:")
print("1. player_visibility.png")
print("2. detection_activity.png")
print("3. track_durations.png")
print(f"\nOutput directory: {OUTPUT_DIR}")
print("============================================")
