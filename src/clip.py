"""clip.py

The clip a run operates on: which video, which frame range, and where every
stage's output for that range lives.

Every entry point resolves one of these first, so the range arithmetic and the
output-naming convention live in exactly one place instead of being copied into
each script.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import cv2

BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_VIDEO = BASE_DIR / "data" / "soccernet" / "england_epl" / "2014-2015" / \
                "2015-02-21 - 18-00 Chelsea 1 - 1 Burnley" / "1_720p.mkv"

# the six stages rebuild_files produces, in order; identity and render are
# optional extras keyed the same way
PIPELINE_STAGES = ("tracking", "keypoint", "team_assignment",
                   "event_detection", "formation", "metrics")


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
        raise ValueError(f"cannot parse position {text!r}")

    value, unit = float(match.group(1)), match.group(2)
    if unit == "m":
        return int(round(value * 60 * fps))
    if unit == "s":
        return int(round(value * fps))
    if unit == "ms":
        return int(round(value * fps / 1000))
    return int(value)  # bare number, or explicit 'f', means frames


@dataclass(frozen=True)
class Clip:
    """A video and an absolute frame range within it."""

    video: Path
    video_id: str
    start: int
    end: int
    fps: float

    @property
    def frames(self) -> int:
        return self.end - self.start

    @property
    def seconds(self) -> float:
        return self.frames / self.fps

    @property
    def name(self) -> str:
        """The stem every stage names its output after."""
        return f"{self.video_id}_{self.start:06d}-{self.end:06d}"

    def stage_path(self, stage: str, suffix: str = ".jsonl") -> Path:
        """Where `stage` keeps its output for this clip."""
        return BASE_DIR / "data" / f"{stage}_output" / f"{self.name}{suffix}"

    def pipeline_paths(self, create_dirs: bool = False) -> dict:
        paths = {stage: self.stage_path(stage) for stage in PIPELINE_STAGES}
        if create_dirs:
            for path in paths.values():
                path.parent.mkdir(parents=True, exist_ok=True)
        return paths

    @classmethod
    def resolve(cls, video, video_id=None, start=None, end=None, duration=None) -> "Clip":
        """Work out the absolute frame range from whatever the caller supplied."""
        video = Path(video)

        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise SystemExit(f"cannot open video: {video}")
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

        return cls(
            video=video,
            video_id=video_id or video.parent.name.replace(" ", "_"),
            start=start_frame,
            end=end_frame,
            fps=fps,
        )

    @staticmethod
    def add_arguments(parser):
        """Add the range flags every entry point shares."""
        parser.add_argument("--video", default=str(DEFAULT_VIDEO), help="path to the match video")
        parser.add_argument("--video-id", default=None, help="name used for the output files")
        parser.add_argument("--start", default=None, help="range start: frames, 90s, 10m or mm:ss")
        parser.add_argument("--end", default=None, help="range end, same formats")
        parser.add_argument("--duration", default=None, help="range length, same formats")
        return parser

    @classmethod
    def from_args(cls, args) -> "Clip":
        return cls.resolve(args.video, video_id=args.video_id, start=args.start,
                           end=args.end, duration=args.duration)
