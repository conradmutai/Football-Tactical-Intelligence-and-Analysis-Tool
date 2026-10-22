"""player_identity.py

Turns tracker ids into player identities.

The tracker gives stable tracklets, but a tracklet still ends at every camera
cut and long occlusion, so one player collects several ids over a clip. This
stage reads the attributes that actually belong to a *person* -- shirt colour
(team), role, and jersey number -- and uses them to merge tracklets.

A note on what each signal can do. Shirt colour only ever gives the team: all
eleven team-mates wear it, so it is a gate that stops cross-team merges, never
an identity. The jersey number is the only real identity signal, and it is
sparse -- legible only on large, rear-facing players, roughly a third of the
crops we bother to try. That is enough because we vote across a whole tracklet
rather than trusting any single frame.

    python -m src.player_identity --video-id smoke --start 20s --duration 30s
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from src.clip import Clip

# A number below this box height is a handful of pixels tall and never readable,
# so attempting OCR there only adds noise.
MIN_CROP_HEIGHT = 55
MAX_CROPS_PER_TRACK = 10

# OCR costs ~200 ms a crop, and a tracklet shorter than this contributes nothing
# to the statistics anyway, so there is no point paying to identify it.
MIN_TRACKLET_FRAMES = 50
MIN_OCR_CONFIDENCE = 0.35
MIN_READS_FOR_NUMBER = 3
MIN_VOTE_SHARE = 0.5

# Squad numbers run 1..MAX_JERSEY_NUMBER. Nobody wears 0, and a read in the 70s
# or 80s on this footage is OCR noise rather than a shirt, so treating those as
# unreadable is more honest than inventing a player.
MIN_JERSEY_NUMBER = 1
MAX_JERSEY_NUMBER = 50

# torso window, as fractions of the box -- the number sits across the back
TORSO_TOP, TORSO_BOTTOM = 0.15, 0.55
TORSO_LEFT, TORSO_RIGHT = 0.15, 0.85
OCR_UPSCALE = 8

MAX_MERGE_GAP_FRAMES = 250  # 10 s -- a camera cut can hide a player this long


def build_tracklets(tracking_path, team_assignment_path):
    """Group detections by track id, with team and role votes attached."""
    teams_by_frame = {}
    with open(team_assignment_path) as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                teams_by_frame[record["frame"]] = {
                    p["track_id"]: p["team"] for p in record["players"]
                }

    tracklets = defaultdict(lambda: {
        "frames": [], "boxes": [], "team_votes": Counter(), "role_votes": Counter(),
    })

    with open(tracking_path) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            frame = record["frame"]
            team_of = teams_by_frame.get(frame, {})

            for detection in record["detections"]:
                track_id = detection["track_id"]
                if track_id is None or detection["class"] == "Ball":
                    continue

                tracklet = tracklets[track_id]
                tracklet["frames"].append(frame)
                tracklet["boxes"].append(detection["bbox"])
                tracklet["role_votes"][detection["class"]] += 1
                if track_id in team_of:
                    tracklet["team_votes"][team_of[track_id]] += 1

    for track_id, tracklet in tracklets.items():
        tracklet["track_id"] = track_id
        tracklet["start"] = min(tracklet["frames"])
        tracklet["end"] = max(tracklet["frames"])
        tracklet["team"] = tracklet["team_votes"].most_common(1)[0][0] if tracklet["team_votes"] else None
        tracklet["role"] = tracklet["role_votes"].most_common(1)[0][0] if tracklet["role_votes"] else "Player"
        tracklet["number"] = None
        tracklet["number_confidence"] = 0.0

    return dict(tracklets)


def _fold_partial_reads(votes):
    """Fold '1' into '10' when the longer read is the one being seen repeatedly.

    OCR often catches only the first digit of a two-digit number, and those
    partials would otherwise outvote the full read.
    """
    folded = Counter(votes)
    for short in [v for v in list(folded) if len(v) == 1]:
        longer = [v for v in folded if len(v) > 1 and v.startswith(short)]
        if not longer:
            continue
        best = max(longer, key=lambda v: folded[v])
        if folded[best] >= folded[short]:
            folded[best] += folded.pop(short)
    return folded


def read_jersey_numbers(video_path, tracklets, reader=None, progress=True):
    """OCR the clearest crops of each tracklet and vote on a jersey number."""
    if reader is None:
        import easyocr  # imported lazily: it loads a model and is slow to start
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    # collect the crops worth trying, keyed by frame so we decode in one pass
    wanted = defaultdict(list)
    skipped = 0
    for track_id, tracklet in tracklets.items():
        if len(tracklet["frames"]) < MIN_TRACKLET_FRAMES:
            skipped += 1
            continue

        candidates = [
            (box[3] - box[1], frame, box)
            for frame, box in zip(tracklet["frames"], tracklet["boxes"])
            if box[3] - box[1] >= MIN_CROP_HEIGHT
        ]
        candidates.sort(reverse=True)
        for _, frame, box in candidates[:MAX_CROPS_PER_TRACK]:
            wanted[frame].append((track_id, box))

    print(f"  reading {sum(len(v) for v in wanted.values())} crops across {len(wanted)} frames "
          f"({skipped} tracklets too short to bother with)", flush=True)

    votes = defaultdict(Counter)
    confidences = defaultdict(list)

    cap = cv2.VideoCapture(str(video_path))
    frame_numbers = sorted(wanted)
    if frame_numbers:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_numbers[0] - 1)

    position = frame_numbers[0] if frame_numbers else 0
    for n, frame_num in enumerate(frame_numbers, 1):
        # step forward to the next wanted frame rather than seeking to it
        frame = None
        while position <= frame_num:
            ok, decoded = cap.read()
            if not ok:
                break
            if position == frame_num:
                frame = decoded
            position += 1
        if frame is None:
            continue

        for track_id, box in wanted[frame_num]:
            x1, y1, x2, y2 = (int(v) for v in box)
            h, w = y2 - y1, x2 - x1
            torso = frame[max(0, y1 + int(TORSO_TOP * h)): y1 + int(TORSO_BOTTOM * h),
                          max(0, x1 + int(TORSO_LEFT * w)): max(0, x1 + int(TORSO_RIGHT * w))]
            if torso.size == 0:
                continue

            big = cv2.resize(torso, None, fx=OCR_UPSCALE, fy=OCR_UPSCALE, interpolation=cv2.INTER_CUBIC)
            for _, digits, confidence in reader.readtext(cv2.cvtColor(big, cv2.COLOR_BGR2GRAY),
                                                         allowlist="0123456789", detail=1):
                digits = digits.strip()
                if (digits and confidence >= MIN_OCR_CONFIDENCE
                        and MIN_JERSEY_NUMBER <= int(digits) <= MAX_JERSEY_NUMBER):
                    votes[track_id][str(int(digits))] += 1
                    confidences[track_id].append(confidence)
                    break

        if progress and n % 200 == 0:
            print(f"  ocr {n}/{len(wanted)} frames", flush=True)

    cap.release()

    for track_id, counter in votes.items():
        counter = _fold_partial_reads(counter)
        number, count = counter.most_common(1)[0]
        total = sum(counter.values())
        share = count / total
        if total >= MIN_READS_FOR_NUMBER and share >= MIN_VOTE_SHARE:
            tracklets[track_id]["number"] = int(number)
            tracklets[track_id]["number_confidence"] = round(share, 2)

    return tracklets


def merge_tracklets(tracklets):
    """Assign a player id to every tracklet, merging those that are one person.

    Tracklets merge when they carry the same confident jersey number and team
    and do not overlap in time -- a player cannot be in two places at once.

    When two tracklets claim the same shirt at the same moment, one of the reads
    must be wrong and we cannot tell which, so they are kept apart under
    distinguishable labels (`A10`, `A10(2)`) rather than being silently fused.

    Tracklets with no legible number keep a synthetic id; there is nothing that
    distinguishes team-mates without one, so merging them would be a guess.
    """
    players = {}
    groups = defaultdict(list)  # (team, number) -> list of non-overlapping groups

    for track_id in sorted(tracklets, key=lambda t: tracklets[t]["start"]):
        tracklet = tracklets[track_id]
        number = tracklet["number"]

        if number is None:
            players[track_id] = f"t{track_id}"
            continue

        key = (tracklet["team"], number)
        shirt = groups[key]

        # join the first group this tracklet can belong to without overlapping it
        target = next((group for group in shirt if not _overlaps(group, tracklet)), None)
        if target is None:
            target = []
            shirt.append(target)

        target.append(tracklet)
        players[track_id] = _player_label(tracklet["team"], number, shirt.index(target))

    return players


def _player_label(team, number, variant):
    base = f"{'AB'[team] if team in (0, 1) else '?'}{number}"
    return base if variant == 0 else f"{base}({variant + 1})"


def _overlaps(group, candidate):
    """True if the candidate shares any frame with a tracklet already in group."""
    return any(not (candidate["end"] < member["start"] or candidate["start"] > member["end"])
               for member in group)


def build_identities(video_path, tracking_path, team_assignment_path, output_path, use_ocr=True):
    tracklets = build_tracklets(tracking_path, team_assignment_path)
    print(f"tracklets: {len(tracklets)}")

    if use_ocr:
        read_jersey_numbers(video_path, tracklets)

    players = merge_tracklets(tracklets)

    numbered = [t for t in tracklets.values() if t["number"] is not None]
    print(f"tracklets with a confident number: {len(numbered)} of {len(tracklets)} "
          f"({100 * len(numbered) / max(len(tracklets), 1):.0f}%)")
    print(f"distinct players after merging: {len(set(players.values()))}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for track_id, tracklet in sorted(tracklets.items()):
            f.write(json.dumps({
                "track_id": track_id,
                "player_id": players[track_id],
                "number": tracklet["number"],
                "number_confidence": tracklet["number_confidence"],
                "team": tracklet["team"],
                "role": tracklet["role"],
                "start_frame": tracklet["start"],
                "end_frame": tracklet["end"],
                "frames": len(tracklet["frames"]),
            }) + "\n")

    print(f"wrote {output_path}")
    for tracklet in sorted(numbered, key=lambda t: -len(t["frames"]))[:15]:
        print(f"  track {tracklet['track_id']:4d}  team {tracklet['team']}  "
              f"#{tracklet['number']:<3d} conf {tracklet['number_confidence']:.2f}  "
              f"{len(tracklet['frames']):4d} frames")

    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    Clip.add_arguments(parser)
    parser.add_argument("--no-ocr", action="store_true", help="skip jersey reading")
    args = parser.parse_args()

    clip = Clip.from_args(args)
    build_identities(
        clip.video,
        clip.stage_path("tracking"),
        clip.stage_path("team_assignment"),
        clip.stage_path("identity"),
        use_ocr=not args.no_ocr,
    )


if __name__ == "__main__":
    main()
