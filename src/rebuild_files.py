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

from src.clip import BASE_DIR, Clip
from src.tracking import tracking
from src.pitch_keypoints import keypoint_detection
from src.team_assignment import assign_teams_for_video
from src.event_detection import run_event_detection, load_joined_positions
from src.formation import classify_windows
from src.metrics import compute_metrics

# COCO-pretrained, used for people and the ball. See the comment in tracking.py
# for why this replaced the custom detector.
PERSON_WEIGHTS = BASE_DIR / "src" / "yolov9m.pt"
# the custom detector, now only used to tag Goalkeeper/Referee
ROLE_WEIGHTS = BASE_DIR / "src" / "runs" / "detect" / "train-7" / "weights" / "best.pt"
POSE_WEIGHTS = BASE_DIR / "src" / "runs" / "pose" / "train-pose-2" / "weights" / "best.pt"


def banner(text):
    print(f"\n{'=' * 60}\n{text}\n{'=' * 60}", flush=True)


def run_pipeline(clip, person_weights=PERSON_WEIGHTS, role_weights=ROLE_WEIGHTS,
                 pose_weights=POSE_WEIGHTS):
    paths = clip.pipeline_paths(create_dirs=True)

    print(f"video      {clip.video}")
    print(f"range      frames {clip.start}..{clip.end}  "
          f"({clip.frames} frames, {clip.seconds:.1f}s at {clip.fps:g} fps)")
    print(f"outputs    data/<stage>_output/{clip.name}.jsonl")

    banner(f"1/6 tracking  ({clip.frames} frames)")
    tracking(clip.video, paths["tracking"], person_weights=person_weights,
             role_weights=role_weights, max_frames=clip.frames, start_frame=clip.start)
    print(f"wrote {paths['tracking']}")

    banner(f"2/6 pitch_keypoints  ({clip.frames} frames)")
    keypoint_detection(clip.video, pose_weights, paths["keypoint"],
                       max_frames=clip.frames, start_frame=clip.start)
    print(f"wrote {paths['keypoint']}")

    banner(f"3/6 team_assignment  ({clip.frames} frames)")
    assign_teams_for_video(clip.video, paths["tracking"], paths["team_assignment"],
                           max_frames=clip.frames, start_frame=clip.start)
    print(f"wrote {paths['team_assignment']}")

    banner("4/6 event_detection")
    run_event_detection(paths["tracking"], paths["team_assignment"], paths["event_detection"])
    with open(paths["event_detection"]) as f:
        windows = [json.loads(line) for line in f]
    print(f"possession windows found: {len(windows)}")
    for window in windows[:5]:
        print(window)

    # Formation and metrics both work in pitch space, so they need the
    # homography join -- which rebuilds a ViewTransformer per frame. Build it
    # once here rather than once per stage.
    banner("joining positions (shared by formation and metrics)")
    joined_positions = load_joined_positions(paths["tracking"], paths["keypoint"],
                                             paths["team_assignment"])
    print(f"frames with usable homography: {len(joined_positions)} of {clip.frames}")

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
        paths["metrics"], joined_positions=joined_positions, fps=clip.fps,
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
    Clip.add_arguments(parser)
    parser.add_argument("--person-weights", default=str(PERSON_WEIGHTS))
    parser.add_argument("--role-weights", default=str(ROLE_WEIGHTS))
    parser.add_argument("--pose-weights", default=str(POSE_WEIGHTS))
    args = parser.parse_args()

    run_pipeline(Clip.from_args(args), person_weights=args.person_weights,
                 role_weights=args.role_weights, pose_weights=args.pose_weights)


if __name__ == "__main__":
    main()
