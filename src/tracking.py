import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from ultralytics import YOLO
from ultralytics.trackers import BOTSORT
from ultralytics.utils import YAML, IterableSimpleNamespace

TRACKER_CONFIG = Path(__file__).resolve().parent.parent / "config" / "tracker.yaml"

# The people detector is a COCO-pretrained model, not the custom one trained on
# the 614-image dataset. The custom detector emits ~30 boxes/frame for ~17 real
# people -- the extras are body fragments (torso, legs) that flicker in and out,
# and those fragments were what manufactured thousands of track IDs. Measured
# over frames 500-1500: next-frame box match 0.856 -> 0.986, distinct IDs
# 695 -> 82, median track lifetime 0.3 s -> 5.9 s. Every box the custom model
# finds overlaps a COCO box, so nothing is lost by the swap.
COCO_PERSON = 0
COCO_SPORTS_BALL = 32

IMGSZ = 1280
CONF = 0.25
NMS_IOU = 0.50
DEVICE = "mps"

# The custom detector still earns its place as a *role tagger*: it is the only
# model that knows Goalkeeper and Referee. Role is constant for a player, so it
# runs every Nth frame and votes per track rather than per frame.
ROLE_EVERY = 5
ROLE_CONF = 0.40
ROLE_OVERLAP = 0.50  # fraction of the role box that must sit inside a person box

# De-duplication thresholds, in pixels at 1280x720. COCO rarely emits the
# fragment boxes this was written for, so it seldom fires -- it is kept as cheap
# insurance against a detector swap reintroducing them, since fragments reaching
# the tracker is what produced thousands of ids.
FOOT_DX = 20  # same player if foot points are this close horizontally...
FOOT_DY = 25  # ...and this close vertically
CONTAINMENT = 0.70  # drop a box with this much of its area inside a better one
FRAGMENT_GAP = 20  # max vertical gap for a stacked box to count as a fragment
FRAGMENT_RATIO = 0.60  # a fragment is this much shorter than the box it belongs to

PERSON_CLASSES = ("Player", "Goalkeeper", "Referee")


def deduplicate_indices(xyxy, conf, classes):
    """Return the indices of boxes to keep, collapsing repeats of one player.

    A detector can emit a full-body box plus a torso and/or legs box for the same
    person. NMS does not remove them -- a torso box inside a body box has an IoU
    of only ~0.4 -- so without this they reach the tracker, where two boxes
    compete for one track and churn its ID.

    Three rules, applied to person classes only (the ball is always kept):
      - foot-point: body and legs boxes both end at the feet, so near-identical
        bottom-centre points mean one player.
      - containment: a box mostly swallowed by a higher-confidence box is part of
        that detection, not a separate person.
      - fragment: a much shorter box stacked directly above or below another,
        for when the body box stopped short of the feet. The height guard keeps
        two genuinely stacked players apart -- one further up the pitch has a
        comparable height, while a fragment is markedly shorter than its parent.

    Args:
        xyxy: (N, 4) array of boxes.
        conf: (N,) array of confidences.
        classes: length-N sequence of class names.

    Returns:
        Sorted list of indices into the input arrays.
    """
    person_idx = [i for i, c in enumerate(classes) if c in PERSON_CLASSES]
    other_idx = [i for i, c in enumerate(classes) if c not in PERSON_CLASSES]

    # highest confidence wins, so ties resolve in favour of the better detection
    person_idx.sort(key=lambda i: conf[i], reverse=True)

    kept = []
    for i in person_idx:
        x1, y1, x2, y2 = xyxy[i]
        foot_x, foot_y = (x1 + x2) / 2.0, y2
        area = max((x2 - x1) * (y2 - y1), 1e-6)
        height = y2 - y1

        duplicate = False
        for j in kept:
            ox1, oy1, ox2, oy2 = xyxy[j]
            other_foot_x, other_foot_y = (ox1 + ox2) / 2.0, oy2

            # rule 1: same feet, same player
            if abs(foot_x - other_foot_x) <= FOOT_DX and abs(foot_y - other_foot_y) <= FOOT_DY:
                duplicate = True
                break

            # rule 2: this box is mostly inside one we already kept
            inter_w = max(0.0, min(x2, ox2) - max(x1, ox1))
            inter_h = max(0.0, min(y2, oy2) - max(y1, oy1))
            if (inter_w * inter_h) / area > CONTAINMENT:
                duplicate = True
                break

            # rule 3: a short fragment stacked against the box it belongs to
            vertical_gap = max(y1 - oy2, oy1 - y2)  # negative when they overlap
            if (
                abs(foot_x - other_foot_x) <= FOOT_DX
                and vertical_gap <= FRAGMENT_GAP
                and height < FRAGMENT_RATIO * (oy2 - oy1)
            ):
                duplicate = True
                break

        if not duplicate:
            kept.append(i)

    return sorted(kept + other_idx)


def build_tracker():
    """Build a BoT-SORT tracker from config/tracker.yaml.

    The tracker is driven directly rather than through `model.track()`, because
    that runs association on the raw detections -- duplicate boxes would reach
    the tracker before they could be removed. Driving it also lets us hand it
    only the person boxes while the ball bypasses tracking entirely.
    """
    cfg = IterableSimpleNamespace(**YAML.load(str(TRACKER_CONFIG)))
    return BOTSORT(cfg)


def _role_hints(role_model, frame, person_xyxy):
    """Map the custom detector's Goalkeeper/Referee boxes onto person boxes.

    Returns {person_index: role_name} for the boxes it can place. Anything it
    does not place stays a Player.
    """
    result = role_model.predict(
        frame, imgsz=IMGSZ, conf=ROLE_CONF, iou=NMS_IOU, device=DEVICE, verbose=False
    )[0]
    boxes = result.boxes.cpu()

    hints = {}
    for box, cls in zip(boxes.xyxy.numpy(), boxes.cls.tolist()):
        name = role_model.names[int(cls)]
        if name not in ("Goalkeeper", "Referee"):
            continue

        area = max((box[2] - box[0]) * (box[3] - box[1]), 1e-6)
        for i, person in enumerate(person_xyxy):
            inter_w = max(0.0, min(box[2], person[2]) - max(box[0], person[0]))
            inter_h = max(0.0, min(box[3], person[3]) - max(box[1], person[1]))
            if (inter_w * inter_h) / area > ROLE_OVERLAP:
                hints[i] = name
                break

    return hints


def tracking(video_path, tracking_path, person_weights, role_weights=None,
             max_frames=None, start_frame=0, role_every=ROLE_EVERY):
    """Detect and track people, writing one JSONL record per frame.

    Frame numbers are absolute and 1-based even when `start_frame` is non-zero,
    because every downstream stage joins on the absolute frame number.
    """
    model = YOLO(str(person_weights))
    role_model = YOLO(str(role_weights)) if role_weights else None
    tracker = build_tracker()

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    idx = start_frame + 1

    # role votes accumulate per track, so a track settles on one role instead of
    # flickering between them on the frames where the role model happens to run
    role_votes = defaultdict(Counter)

    with open(str(tracking_path), "w") as f:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            result = model.predict(
                frame, imgsz=IMGSZ, conf=CONF, iou=NMS_IOU, device=DEVICE,
                classes=[COCO_PERSON, COCO_SPORTS_BALL], verbose=False,
            )[0]

            # the tracker converts boxes to numpy internally, so move off MPS here
            boxes = result.boxes.cpu()
            cls = boxes.cls.numpy()
            person_sel = np.flatnonzero(cls == COCO_PERSON)
            ball_sel = np.flatnonzero(cls == COCO_SPORTS_BALL)

            person_boxes = boxes[person_sel]
            keep = deduplicate_indices(
                person_boxes.xyxy.numpy(),
                person_boxes.conf.numpy(),
                ["Player"] * len(person_sel),
            )
            person_boxes = person_boxes[keep]

            # the frame is passed so BoT-SORT can run global motion compensation,
            # which cancels the broadcast camera's pan before matching on IoU
            tracks = tracker.update(person_boxes, frame)

            if role_model is not None and tracks.size and (idx - 1) % role_every == 0:
                for i, role in _role_hints(role_model, frame, tracks[:, :4]).items():
                    role_votes[int(tracks[i, 4])][role] += 1

            detections = []
            for row in tracks:
                x1, y1, x2, y2, track_id, score = row[:6]
                track_id = int(track_id)
                votes = role_votes.get(track_id)
                detections.append({
                    "track_id": track_id,
                    "class": votes.most_common(1)[0][0] if votes else "Player",
                    "bbox": [round(float(v), 1) for v in (x1, y1, x2, y2)],
                    "confidence": round(float(score), 3),
                })

            # the ball needs a position, not an identity, so it skips the tracker
            for i in ball_sel:
                detections.append({
                    "track_id": None,
                    "class": "Ball",
                    "bbox": [round(float(v), 1) for v in boxes.xyxy.numpy()[i]],
                    "confidence": round(float(boxes.conf.numpy()[i]), 3),
                })

            f.write(json.dumps({"frame": idx, "timestamp": idx / fps, "detections": detections}) + "\n")

            idx += 1
            if max_frames is not None and idx > start_frame + max_frames:
                break

    cap.release()
