"""render_video.py

Renders the pipeline's output as a video: the annotated broadcast frame, a
top-down pitch radar, and a live statistics sidebar.

    python -m src.render_video --video-id smoke --start 20s --duration 30s

Runs no inference. Everything comes from the JSONL the pipeline already wrote,
so re-rendering is cheap and can be iterated on freely.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.configs.soccer import SoccerPitchConfiguration

from src.homography import build_transformer_for_frame
from src.clip import Clip

# --- layout -----------------------------------------------------------------
CANVAS_W, CANVAS_H = 1920, 1080
BROADCAST_W, BROADCAST_H = 1280, 720
RADAR_W, RADAR_H = 1280, CANVAS_H - BROADCAST_H  # 1280 x 360
SIDEBAR_X, SIDEBAR_W = BROADCAST_W, CANVAS_W - BROADCAST_W  # 640 wide

PITCH_CONFIG = SoccerPitchConfiguration()
RADAR_PADDING = 20
# fit the 12000x7000 pitch inside the radar strip
RADAR_SCALE = min((RADAR_W - 4 * RADAR_PADDING) / PITCH_CONFIG.length,
                  (RADAR_H - 2 * RADAR_PADDING) / PITCH_CONFIG.width)

# a homography is only available on roughly a third of frames; carrying the last
# good one over a short gap fills in camera wobble without inventing geometry
MAX_HOMOGRAPHY_CARRY = 25

# --- palette (BGR) ----------------------------------------------------------
BG = (24, 22, 20)
PANEL = (38, 35, 32)
TEXT = (235, 235, 235)
MUTED = (150, 148, 145)
ACCENT = (90, 200, 255)
TEAM_COLOURS = [(235, 170, 60), (70, 70, 235)]  # fallback; real kit colours are sampled
REFEREE_COLOUR = (60, 220, 255)

# radar groups are the two team indices plus this sentinel, rather than mixing
# ints and the bare string "referee" as dict keys
REFEREE_GROUP = "referee"
BALL_COLOUR = (255, 255, 255)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def sample_team_colours(video_path, tracking_path, teams, start_frame, sample_frames=40):
    """Take each team's display colour from the shirts themselves.

    The team labels are bare KMeans cluster ids with no colour attached, so a
    fixed palette ends up drawing Chelsea in red. Sampling the actual torso
    pixels keeps the overlay readable against the broadcast.
    """
    frames_with_players = [f for f in sorted(teams) if teams[f]["players"]]
    if not frames_with_players:
        return list(TEAM_COLOURS)

    step = max(1, len(frames_with_players) // sample_frames)
    wanted = set(frames_with_players[::step][:sample_frames])

    boxes = {}
    for record in stream_jsonl(tracking_path):
        if record["frame"] in wanted:
            boxes[record["frame"]] = record["detections"]

    cap = cv2.VideoCapture(str(video_path))
    swatches = defaultdict(list)
    for frame_num in sorted(wanted):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
        ok, frame = cap.read()
        if not ok:
            continue
        team_of = {p["track_id"]: p["team"] for p in teams[frame_num]["players"]}
        for detection in boxes.get(frame_num, []):
            team = team_of.get(detection["track_id"])
            if team not in (0, 1) or detection["class"] != "Player":
                continue
            x1, y1, x2, y2 = (int(v) for v in detection["bbox"])
            h, w = y2 - y1, x2 - x1
            # match TeamAssigner's torso window, sides trimmed to avoid grass
            torso = frame[max(0, y1 + int(0.2 * h)):y1 + int(0.55 * h),
                          max(0, x1 + int(0.25 * w)):max(0, x1 + int(0.75 * w))]
            if torso.size:
                swatches[team].append(np.median(torso.reshape(-1, 3), axis=0))
    cap.release()

    colours = list(TEAM_COLOURS)
    for team in (0, 1):
        if swatches[team]:
            bgr = np.median(np.array(swatches[team]), axis=0).astype(np.uint8)
            # keep the hue -- that is what tells the kits apart -- but force
            # enough saturation and brightness to read as an overlay
            hsv = cv2.cvtColor(bgr.reshape(1, 1, 3), cv2.COLOR_BGR2HSV)[0, 0].astype(int)
            hsv[1] = max(hsv[1], 180)
            hsv[2] = max(hsv[2], 210)
            lifted = cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2BGR)[0, 0]
            colours[team] = tuple(int(v) for v in lifted)
    return colours


def load_jsonl(path, key):
    records = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records[record[key]] = record
    return records


def stream_jsonl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def text(img, label, x, y, scale=0.5, colour=TEXT, weight=1):
    cv2.putText(img, label, (x, y), FONT, scale, colour, weight, cv2.LINE_AA)


def draw_radar(player_points, ball_point, stale, team_colours):
    """Top-down pitch with the current positions, or a notice when unavailable."""
    panel = np.full((RADAR_H, RADAR_W, 3), PANEL, dtype=np.uint8)

    if player_points is None:
        text(panel, "NO PITCH REGISTRATION", RADAR_W // 2 - 150, RADAR_H // 2, 0.7, MUTED, 2)
        text(panel, "pitch lines not visible in this shot",
             RADAR_W // 2 - 150, RADAR_H // 2 + 26, 0.45, MUTED)
        return panel

    pitch = draw_pitch(PITCH_CONFIG, padding=RADAR_PADDING, scale=RADAR_SCALE,
                       line_thickness=1, point_radius=3)

    for group, points in player_points.items():
        if not points:
            continue
        colour = REFEREE_COLOUR if group is REFEREE_GROUP else team_colours[group]
        pitch = draw_points_on_pitch(
            PITCH_CONFIG, np.array(points), face_color=sv.Color(*colour[::-1]),
            edge_color=sv.Color.BLACK, radius=7, thickness=1,
            padding=RADAR_PADDING, scale=RADAR_SCALE, pitch=pitch,
        )

    if ball_point is not None:
        pitch = draw_points_on_pitch(
            PITCH_CONFIG, np.array([ball_point]), face_color=sv.Color.WHITE,
            edge_color=sv.Color.BLACK, radius=5, thickness=1,
            padding=RADAR_PADDING, scale=RADAR_SCALE, pitch=pitch,
        )

    # centre the pitch in the strip
    h, w = pitch.shape[:2]
    h, w = min(h, RADAR_H), min(w, RADAR_W)
    y0, x0 = (RADAR_H - h) // 2, (RADAR_W - w) // 2
    panel[y0:y0 + h, x0:x0 + w] = pitch[:h, :w]

    if stale:
        text(panel, "holding last registration", 12, RADAR_H - 12, 0.42, MUTED)

    return panel


def draw_sidebar(state, team_colours):
    """Statistics panel: clock, possession, formation, distances, tracking health."""
    panel = np.full((CANVAS_H, SIDEBAR_W, 3), PANEL, dtype=np.uint8)
    pad = 24
    y = 54

    text(panel, state["title"], pad, y, 0.78, TEXT, 2)
    y += 30
    text(panel, f"{state['clock']}   frame {state['frame']}", pad, y, 0.5, MUTED)
    y += 40
    cv2.line(panel, (pad, y), (SIDEBAR_W - pad, y), (70, 66, 62), 1)
    y += 38

    # --- possession
    text(panel, "POSSESSION", pad, y, 0.52, ACCENT, 1)
    y += 30
    bar_w = SIDEBAR_W - 2 * pad
    for team in (0, 1):
        share = state["possession"][team]
        cv2.rectangle(panel, (pad, y - 11), (pad + 14, y + 1), team_colours[team], -1)
        text(panel, state["team_names"][team], pad + 22, y, 0.5, TEXT)
        text(panel, f"{share * 100:4.1f}%", SIDEBAR_W - pad - 58, y, 0.5, TEXT)
        y += 12
        cv2.rectangle(panel, (pad, y), (pad + bar_w, y + 12), (58, 55, 52), -1)
        if share > 0:
            cv2.rectangle(panel, (pad, y), (pad + int(bar_w * share), y + 12),
                          team_colours[team], -1)
        y += 34

    y += 6
    holder = state["in_possession"]
    text(panel, "IN POSSESSION NOW", pad, y, 0.46, MUTED)
    y += 26
    if holder is None:
        text(panel, "contested / no ball", pad, y, 0.58, MUTED)
    else:
        text(panel, state["team_names"][holder], pad, y, 0.62, team_colours[holder], 2)
    y += 40
    cv2.line(panel, (pad, y), (SIDEBAR_W - pad, y), (70, 66, 62), 1)
    y += 38

    # --- formation
    text(panel, f"FORMATION  (window {state['window']})" if state["window"] else "FORMATION",
         pad, y, 0.52, ACCENT, 1)
    y += 30
    for label, value in state["formations"]:
        text(panel, label, pad, y, 0.48, MUTED)
        text(panel, value, pad + 150, y, 0.56, TEXT)
        y += 28

    y += 14
    cv2.line(panel, (pad, y), (SIDEBAR_W - pad, y), (70, 66, 62), 1)
    y += 38

    # --- distances
    text(panel, "DISTANCE THIS WINDOW", pad, y, 0.52, ACCENT, 1)
    y += 30
    if not state["top_distance"]:
        text(panel, "no window data", pad, y, 0.48, MUTED)
        y += 26
    for label, metres, colour in state["top_distance"]:
        text(panel, label, pad, y, 0.5, colour)
        text(panel, f"{metres:5.1f} m", SIDEBAR_W - pad - 70, y, 0.5, TEXT)
        y += 26

    # --- tracking health, pinned to the bottom
    y = CANVAS_H - 150
    cv2.line(panel, (pad, y), (SIDEBAR_W - pad, y), (70, 66, 62), 1)
    y += 32
    text(panel, "TRACKING", pad, y, 0.52, ACCENT, 1)
    y += 28
    for label, value, ok in state["health"]:
        text(panel, label, pad, y, 0.46, MUTED)
        text(panel, value, pad + 210, y, 0.46, TEXT if ok else (90, 90, 220))
        y += 24

    return panel


def possession_by_frame(windows):
    """frame -> (window_num, team), so the render loop does an O(1) lookup.

    Scanning the window list per frame was 158 x 15,000 comparisons for
    something the data can answer directly.
    """
    lookup = {}
    for window in windows.values():
        for frame in range(window["start_frame"], window["end_frame"] + 1):
            lookup[frame] = (window["window_num"], window["team_in_possession"])
    return lookup


def player_label(track_id, identities):
    """A bare number is a real shirt number; "#" marks a tracker id."""
    identity = identities.get(track_id)
    if identity and identity.get("number") is not None:
        return str(identity["number"])
    return f"#{track_id}"


def render(clip, out_path, team_names, identity_path=None):
    paths = clip.pipeline_paths()
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise SystemExit("missing pipeline output:\n  " + "\n  ".join(missing))

    # the identity stage is optional; without it the overlay falls back to raw
    # track ids, which still read correctly, just not as shirt numbers
    if identity_path is None and clip.stage_path("identity").exists():
        identity_path = clip.stage_path("identity")

    teams = load_jsonl(paths["team_assignment"], "frame")
    windows = load_jsonl(paths["event_detection"], "window_num")
    formations = load_jsonl(paths["formation"], "window")
    metrics = load_jsonl(paths["metrics"], "window")
    transformers = build_transformer_for_frame(paths["keypoint"])
    possession = possession_by_frame(windows)
    team_colours = sample_team_colours(clip.video, paths["tracking"], teams, clip.start)
    for team in (0, 1):
        b, g, r = team_colours[team]
        print(f"  cluster {team} -> {team_names[team]:12s} kit colour #{r:02x}{g:02x}{b:02x}")

    identities = {}
    if identity_path and Path(identity_path).exists():
        for record in stream_jsonl(identity_path):
            identities[record["track_id"]] = record
        named = sum(1 for r in identities.values() if r.get("number") is not None)
        print(f"  identities: {named} of {len(identities)} tracklets carry a shirt number")

    cap = cv2.VideoCapture(str(clip.video))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"avc1"),
                             clip.fps, (CANVAS_W, CANVAS_H))
    if not writer.isOpened():
        raise SystemExit(f"could not open video writer for {out_path}")

    possession_frames = defaultdict(int)
    last_transformer = None
    last_transformer_frame = None
    written = 0
    position = clip.start  # frames consumed from the decoder so far

    for record in stream_jsonl(paths["tracking"]):
        frame_num = record["frame"]

        # keep the decoder locked to the record being drawn. Reading blindly in
        # step would silently paint overlays onto the wrong frame if the
        # tracking output ever had a gap.
        if frame_num != position + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
            position = frame_num - 1

        success, frame = cap.read()
        if not success:
            break
        position += 1

        canvas = np.full((CANVAS_H, CANVAS_W, 3), BG, dtype=np.uint8)
        broadcast = frame.copy()

        team_of = {p["track_id"]: p["team"] for p in teams.get(frame_num, {}).get("players", [])}

        window_num, in_possession = possession.get(frame_num, (None, None))
        if in_possession is not None:
            possession_frames[in_possession] += 1
        total_possession = sum(possession_frames.values()) or 1

        # --- homography, with a short carry across gaps
        transformer = transformers.get(frame_num)
        if transformer is not None:
            last_transformer, last_transformer_frame = transformer, frame_num
        stale = False
        if (transformer is None and last_transformer_frame is not None
                and frame_num - last_transformer_frame <= MAX_HOMOGRAPHY_CARRY):
            transformer, stale = last_transformer, True

        radar_points = defaultdict(list)
        ball_point = None

        # --- annotate the broadcast frame
        for detection in record["detections"]:
            x1, y1, x2, y2 = detection["bbox"]
            cls = detection["class"]
            track_id = detection["track_id"]

            if cls == "Ball":
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                cv2.circle(broadcast, (cx, cy), 7, BALL_COLOUR, 2, cv2.LINE_AA)
                cv2.circle(broadcast, (cx, cy), 2, BALL_COLOUR, -1, cv2.LINE_AA)
                if transformer is not None:
                    ball_point = transformer.transform_points(np.array([[cx, cy]]))[0]
                continue

            if cls == "Referee":
                colour, group, label = REFEREE_COLOUR, REFEREE_GROUP, "REF"
            else:
                team = team_of.get(track_id)
                colour = team_colours[team] if team in (0, 1) else MUTED
                group = team if team in (0, 1) else None
                label = player_label(track_id, identities)
                if cls == "Goalkeeper":
                    label += " GK"

            # ellipse at the feet reads better than a box over a moving player
            foot = (int((x1 + x2) / 2), int(y2))
            axis = max(8, int((x2 - x1) * 0.55))
            cv2.ellipse(broadcast, foot, (axis, max(4, axis // 3)), 0, -30, 210,
                        colour, 2, cv2.LINE_AA)

            tw = cv2.getTextSize(label, FONT, 0.42, 1)[0][0]
            cv2.rectangle(broadcast, (foot[0] - tw // 2 - 4, foot[1] + 4),
                          (foot[0] + tw // 2 + 4, foot[1] + 20), colour, -1)
            text(broadcast, label, foot[0] - tw // 2, foot[1] + 16, 0.42, (20, 20, 20))

            if transformer is not None and group is not None:
                radar_points[group].append(transformer.transform_points(np.array([foot]))[0])

        canvas[0:BROADCAST_H, 0:BROADCAST_W] = broadcast
        canvas[BROADCAST_H:CANVAS_H, 0:RADAR_W] = draw_radar(
            radar_points if transformer is not None else None, ball_point, stale, team_colours)

        # --- sidebar state
        formation = formations.get(window_num, {})
        metric = metrics.get(window_num, {})
        top = []
        for entry in sorted(metric.get("window_metrics", []),
                            key=lambda m: -m["distance_covered"])[:5]:
            team = team_of.get(entry["track_id"])
            top.append((player_label(entry["track_id"], identities),
                        entry["distance_covered"] / 100.0,
                        team_colours[team] if team in (0, 1) else MUTED))

        seconds = frame_num / clip.fps
        people = sum(1 for d in record["detections"] if d["class"] != "Ball")
        state = {
            "title": f"{team_names[0]} vs {team_names[1]}",
            "clock": f"{int(seconds // 60):02d}:{seconds % 60:05.2f}",
            "frame": frame_num,
            "team_names": team_names,
            "possession": [possession_frames[0] / total_possession,
                           possession_frames[1] / total_possession],
            "in_possession": in_possession,
            "window": window_num,
            "formations": [
                ("attacking", str(formation.get("offensive_formation", ["-"])[0])),
                ("defending", str(formation.get("defensive_formation", ["-"])[0])),
            ],
            "top_distance": top,
            "health": [
                ("people tracked", str(people), people >= 10),
                ("ball detected",
                 "yes" if any(d["class"] == "Ball" for d in record["detections"]) else "no", True),
                ("pitch registration",
                 "locked" if not stale and transformer is not None else
                 "held" if stale else "none", transformer is not None),
            ],
        }
        canvas[0:CANVAS_H, SIDEBAR_X:CANVAS_W] = draw_sidebar(state, team_colours)

        writer.write(canvas)
        written += 1

    writer.release()
    cap.release()
    print(f"wrote {out_path}  ({written} frames, {written / clip.fps:.1f}s)")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    Clip.add_arguments(parser)
    parser.add_argument("--out", default=None)
    parser.add_argument("--identity", default=None, help="optional identity_output JSONL")
    parser.add_argument("--team-names", default="Team A,Team B")
    parser.add_argument("--swap-teams", action="store_true",
                        help="the two colour clusters come out in an arbitrary order; "
                             "use this if the names landed on the wrong kits")
    args = parser.parse_args()

    clip = Clip.from_args(args)

    names = [n.strip() for n in args.team_names.split(",")][:2]
    while len(names) < 2:
        names.append(f"Team {len(names) + 1}")
    if args.swap_teams:
        names.reverse()

    out = Path(args.out) if args.out else clip.stage_path("render", ".mp4")
    render(clip, out, names, args.identity)


if __name__ == "__main__":
    main()
