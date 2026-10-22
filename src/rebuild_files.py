"""rebuild_files.py

Runs the whole pipeline (tracking -> keypoints -> team_assignment ->
event_detection -> formation -> metrics) over an arbitrary frame range of an
arbitrary video.

    python -m src.rebuild_files --video <path> --start 0 --duration 10m

Ranges accept frames (`1500`), seconds (`90s`), minutes (`10m`) or a clock
(`mm:ss`). Frame numbers written to the JSONL are always absolute, so a range
starting at 5m joins correctly against anything else from the same video.

The expensive part is YOLO inference across the range, not the downstream
stages.
"""

import argparse
import json
import re
from pathlib import Path

import cv2

from src.tracking import tracking
from src.pitch_keypoints import keypoint_detection
from src.team_assignment import assign_teams_for_video
from src.event_detection import run_event_detection, load_joined_positions
from src.formation import classify_windows
from src.metrics import compute_metrics

BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_VIDEO = BASE_DIR / "data" / "soccernet" / "england_epl" / "2014-2015" / \
                "2015-02-21 - 18-00 Chelsea 1 - 1 Burnley" / "1_720p.mkv"

# COCO-pretrained, used for people and the ball. See the comment in tracking.py
# for why this replaced the custom detector.
PERSON_WEIGHTS = BASE_DIR / "src" / "yolov9m.pt"
# the custom detector, now only used to tag Goalkeeper/Referee
ROLE_WEIGHTS = BASE_DIR / "src" / "runs" / "detect" / "train-7" / "weights" / "best.pt"
POSE_WEIGHTS = BASE_DIR / "src" / "runs" / "pose" / "train-pose-2" / "weights" / "best.pt"

STAGES = ("tracking", "keypoint", "team_assignment", "event_detection", "formation", "metrics")


def parse_position(text, fps):
    """Turn '1500', '90s', '10m' or '2:30' into a frame number."""
    if text is None:
        return None

    text = str(text).strip().lower()

    if ":" in text:
        minutes, seconds = text.split(":")
        return int(round((int(minutes) * 60 + float(seconds)) * fps))

    match = re.fullmatch(r"(\d+(?:\.\d+)?)(ms|s|m|f)?", text)
    if not match:
        raise argparse.ArgumentTypeError(f"cannot parse position {text!r}")

    value, unit = float(match.group(1)), match.group(2)
    if unit == "m":
        return int(round(value * 60 * fps))
    if unit == "s":
        return int(round(value * fps))
    if unit == "ms":
        return int(round(value * fps / 1000))
    return int(value)  # bare number, or explicit 'f', means frames


def output_paths(video_id, start_frame, end_frame):
    """One file per stage, with the range in the name so runs cannot clobber."""
    suffix = f"{video_id}_{start_frame:06d}-{end_frame:06d}"
    paths = {}
    for stage in STAGES:
        path = BASE_DIR / "data" / f"{stage}_output" / f"{suffix}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        paths[stage] = path
    return paths


def banner(text):
    print(f"\n{'=' * 60}\n{text}\n{'=' * 60}", flush=True)


def run_pipeline(video_path, video_id=None, start=None, end=None, duration=None,
                 person_weights=PERSON_WEIGHTS, role_weights=ROLE_WEIGHTS, pose_weights=POSE_WEIGHTS):
    video_path = Path(video_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    start_frame = parse_position(start, fps) or 0
    if duration is not None:
        end_frame = start_frame + parse_position(duration, fps)
    elif end is not None:
        end_frame = parse_position(end, fps)
    else:
        end_frame = total_frames
    end_frame = min(end_frame, total_frames)

    if end_frame <= start_frame:
        raise SystemExit(f"empty range: {start_frame}..{end_frame}")

    max_frames = end_frame - start_frame
    video_id = video_id or video_path.parent.name.replace(" ", "_")
    paths = output_paths(video_id, start_frame, end_frame)

    print(f"video      {video_path}")
    print(f"range      frames {start_frame}..{end_frame}  "
          f"({max_frames} frames, {max_frames / fps:.1f}s at {fps:g} fps)")
    print(f"outputs    data/<stage>_output/{paths['tracking'].name}")

    banner(f"1/6 tracking  ({max_frames} frames)")
    tracking(video_path, paths["tracking"], person_weights=person_weights,
             role_weights=role_weights, max_frames=max_frames, start_frame=start_frame)
    print(f"wrote {paths['tracking']}")

    banner(f"2/6 pitch_keypoints  ({max_frames} frames)")
    keypoint_detection(video_path, pose_weights, paths["keypoint"],
                       max_frames=max_frames, start_frame=start_frame)
    print(f"wrote {paths['keypoint']}")

    banner(f"3/6 team_assignment  ({max_frames} frames)")
    assign_teams_for_video(video_path, paths["tracking"], paths["team_assignment"],
                           max_frames=max_frames, start_frame=start_frame)
    print(f"wrote {paths['team_assignment']}")

    # The tracking/keypoint/team join rebuilds a ViewTransformer for every frame.
    # Stages 4, 5 and 6 all need it, so build it once here instead of three times.
    banner("joining positions (shared by stages 4-6)")
    joined_positions = load_joined_positions(paths["tracking"], paths["keypoint"], paths["team_assignment"])
    print(f"frames with usable homography: {len(joined_positions)} of {max_frames}")

    banner("4/6 event_detection")
    run_event_detection(paths["tracking"], paths["keypoint"], paths["team_assignment"],
                        paths["event_detection"], joined_positions=joined_positions)
    with open(paths["event_detection"]) as f:
        windows = [json.loads(line) for line in f]
    print(f"possession windows found: {len(windows)}")
    for window in windows[:5]:
        print(window)

    banner("5/6 formation")
    formation_results = classify_windows(
        paths["event_detection"], paths["tracking"], paths["keypoint"], paths["team_assignment"],
        paths["formation"], joined_positions=joined_positions,
    )
    print(f"windows classified: {len(formation_results)}")
    for result in formation_results[:5]:
        print(result)

    banner("6/6 metrics")
    compute_metrics(
        paths["event_detection"], paths["tracking"], paths["keypoint"], paths["team_assignment"],
        paths["metrics"], joined_positions=joined_positions, fps=fps,
    )
    with open(paths["metrics"]) as f:
        metrics_windows = [json.loads(line) for line in f]
    print(f"windows with metrics: {len(metrics_windows)}")

    banner("done")
    for stage, path in paths.items():
        print(f"{stage:18s} {path}")

    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", default=str(DEFAULT_VIDEO), help="path to the match video")
    parser.add_argument("--video-id", default=None, help="name used for the output files")
    parser.add_argument("--start", default=None, help="range start: frames, 90s, 10m or mm:ss")
    parser.add_argument("--end", default=None, help="range end, same formats")
    parser.add_argument("--duration", default=None, help="range length, same formats")
    parser.add_argument("--person-weights", default=str(PERSON_WEIGHTS))
    parser.add_argument("--role-weights", default=str(ROLE_WEIGHTS))
    parser.add_argument("--pose-weights", default=str(POSE_WEIGHTS))
    args = parser.parse_args()

    run_pipeline(args.video, video_id=args.video_id, start=args.start, end=args.end,
                 duration=args.duration, person_weights=args.person_weights,
                 role_weights=args.role_weights, pose_weights=args.pose_weights)


if __name__ == "__main__":
    main()
