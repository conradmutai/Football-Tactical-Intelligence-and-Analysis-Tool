"""
rebuild_pipeline.py

Regenerates every JSONL in the pipeline (tracking -> keypoints ->
team_assignment -> event_detection -> formation -> metrics) at a shared,
larger frame count.

Run this from the project root (same convention as your other
if __name__ == "__main__" test blocks). Expect this to take a while —
YOLO inference + keypoint inference across MAX_FRAMES frames is the
real cost here, not the downstream stages.
"""

import json
from pathlib import Path

from src.tracking import tracking
from src.pitch_keypoints import keypoint_detection
from src.team_assignment import assign_teams_for_video
from src.event_detection import run_event_detection
from src.formation import classify_windows
from src.metrics import compute_metrics

# --- shared config -----------------------------------------------------
MAX_FRAMES = 15000  # ~10 min at 25fps — raise/lower as needed

VIDEO_ID = "2015-02-21_chelsea_burnley"
BASE_DIR = Path(__file__).resolve().parent.parent

VIDEO_PATH = BASE_DIR / "data" / "soccernet" / "england_epl" / "2014-2015" / \
             "2015-02-21 - 18-00 Chelsea 1 - 1 Burnley" / "1_720p.mkv"

DETECTION_WEIGHTS = Path(__file__).resolve().parent / "runs" / "detect" / "train-7" / "weights" / "best.pt"
POSE_WEIGHTS = Path(__file__).resolve().parent / "runs" / "pose" / "train-pose-2" / "weights" / "best.pt"

TRACKING_OUTPUT_PATH = BASE_DIR / "data" / "tracking_output" / f"{VIDEO_ID}.jsonl"
KEYPOINT_OUTPUT_PATH = BASE_DIR / "data" / "keypoint_output" / f"{VIDEO_ID}.jsonl"
TEAM_ASSIGNMENT_OUTPUT_PATH = BASE_DIR / "data" / "team_assignment_output" / f"{VIDEO_ID}.jsonl"
EVENT_DETECTION_OUTPUT_PATH = BASE_DIR / "data" / "event_detection_output" / f"{VIDEO_ID}.jsonl"
FORMATION_OUTPUT_PATH = BASE_DIR / "data" / "formation_output" / f"{VIDEO_ID}.jsonl"
METRICS_OUTPUT_PATH = BASE_DIR / "data" / "metrics_output" / f"{VIDEO_ID}.jsonl"

# make sure output dirs exist — none of the stage functions create them
for path in (TRACKING_OUTPUT_PATH, KEYPOINT_OUTPUT_PATH, TEAM_ASSIGNMENT_OUTPUT_PATH,
             EVENT_DETECTION_OUTPUT_PATH, FORMATION_OUTPUT_PATH, METRICS_OUTPUT_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)


def stage(name):
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")


if __name__ == "__main__":
    stage(f"1/6 tracking.py  (max_frames={MAX_FRAMES})")
    tracking(DETECTION_WEIGHTS, VIDEO_PATH, TRACKING_OUTPUT_PATH, max_frames=MAX_FRAMES)
    print(f"wrote {TRACKING_OUTPUT_PATH}")

    stage(f"2/6 pitch_keypoints.py  (max_frames={MAX_FRAMES})")
    keypoint_detection(VIDEO_PATH, POSE_WEIGHTS, KEYPOINT_OUTPUT_PATH, max_frames=MAX_FRAMES)
    print(f"wrote {KEYPOINT_OUTPUT_PATH}")

    stage(f"3/6 team_assignment.py  (max_frames={MAX_FRAMES})")
    assign_teams_for_video(VIDEO_PATH, TRACKING_OUTPUT_PATH, TEAM_ASSIGNMENT_OUTPUT_PATH, max_frames=MAX_FRAMES)
    print(f"wrote {TEAM_ASSIGNMENT_OUTPUT_PATH}")

    stage("4/6 event_detection.py")
    run_event_detection(TRACKING_OUTPUT_PATH, KEYPOINT_OUTPUT_PATH, TEAM_ASSIGNMENT_OUTPUT_PATH, EVENT_DETECTION_OUTPUT_PATH)

    with open(EVENT_DETECTION_OUTPUT_PATH) as f:
        windows = [json.loads(line) for line in f]
    print(f"possession windows found: {len(windows)}")
    for w in windows[:5]:
        print(w)

    stage("5/6 formation.py")
    formation_results = classify_windows(
        EVENT_DETECTION_OUTPUT_PATH, TRACKING_OUTPUT_PATH, KEYPOINT_OUTPUT_PATH, TEAM_ASSIGNMENT_OUTPUT_PATH,
        FORMATION_OUTPUT_PATH
    )
    print(f"windows classified: {len(formation_results)}")
    for r in formation_results[:5]:
        print(r)

    stage("6/6 metrics.py")
    compute_metrics(
        EVENT_DETECTION_OUTPUT_PATH, TRACKING_OUTPUT_PATH, KEYPOINT_OUTPUT_PATH, TEAM_ASSIGNMENT_OUTPUT_PATH,
        METRICS_OUTPUT_PATH
    )

    with open(METRICS_OUTPUT_PATH) as f:
        metrics_windows = [json.loads(line) for line in f]
    print(f"windows with metrics: {len(metrics_windows)}")
    for m in metrics_windows[:2]:
        print(m)