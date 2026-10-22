import json
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import numpy as np

from src.homography import build_transformer_for_frame

# Possession is a question about who is nearest the ball, which does not need a
# pitch model -- and tying it to one cost most of the data: the homography
# resolves on ~32% of frames while the ball is seen on ~64%, so pitch-space
# possession could only ever see their 19% intersection. Measuring in image
# space instead uses every frame that has a ball.
PLAYER_HEIGHT_M = 1.8  # a player's bounding box is the ruler for image distance
POSSESSION_RADIUS_M = 1.5
DUEL_RADIUS_M = 2.5


# euclidian distance calculation
def euclidian_distance(point_a: np.ndarray, point_b: np.ndarray) -> float:
    x_diff = point_b[0] - point_a[0]
    y_diff = point_b[1] - point_a[1]
    dist = np.sqrt(np.square(y_diff) + np.square(x_diff))
    return dist


# finds the nearest player to the ball
def find_nearest_player(ball_position: np.ndarray, player_position: dict) -> Optional[List]:
    # initializing the return
    nearest_id = None
    min_dist = float('inf')

    # iterates through all player positions
    for track_id, player_pos in player_position.items():
        dist = euclidian_distance(ball_position, player_pos)

        # compares the distances and choosing the min distance
        if dist < min_dist:
            min_dist = dist
            nearest_id = track_id

    nearest_player = [nearest_id, min_dist]

    return nearest_player


# detects the possession type per frame
def detect_possession_for_frame(ball_pos: Optional[np.ndarray], player_positions: dict, player_teams: dict, r_possession: float,
                                r_duel: float) -> dict:
    # checks if ball is out of frame and returns an empty dict
    if ball_pos is None:
        record = {
            "track_id": None,
            "status": "No Ball Detected",
            "min_distance": None,
            "team": None
        }

        return record

    # checks who's nearest to the ball and their distance to the ball
    nearest_id, min_dist = find_nearest_player(ball_pos, player_positions)

    # checks the ball to nearest player radius to determine the status of the ball in play
    if min_dist <= r_possession:
        status = "possession"
    elif min_dist <= r_duel:
        status = "in-duel"
    else:
        status = "no-one in possession"

    team = player_teams.get(nearest_id)

    # whoever is nearest has to belong to a team for this to be possession at
    # all -- a referee or an unlabelled track cannot hold the ball for anyone
    if team is None:
        status = "no-one in possession"

    # creates a dict
    record = {
        "track_id": nearest_id,
        "status": status,
        "min_distance": min_dist,
        "team": team
    }

    return record


def build_possession_windows(per_frame_possession: List[Optional[dict]], min_duration_frames: int = 10) -> dict:
    # creates return record
    record = defaultdict()

    window = 1  # initializing all to 1

    # assign a team. A short clip can contain no clean-possession frame at all
    # (a replay, a crowd shot), so this must return empty rather than raise.
    first_valid = next(
        (fp for fp in per_frame_possession
         if fp["status"] not in ("no-one in possession", "in-duel", "No Ball Detected")
         and fp["team"] is not None),
        None
    )
    if first_valid is None:
        return record

    window_start = first_valid["frame"]
    team = first_valid["team"]

    last_frame_num = None

    for frame_possession in per_frame_possession:
        # skips frames where no one is in possession or when in duel
        if (frame_possession["status"] in ("no-one in possession", "in-duel", "No Ball Detected")
                or frame_possession["team"] is None):
            continue

        # initializes factors for checks
        current_team = frame_possession["team"]
        frame_num = frame_possession["frame"]
        duration = frame_num - window_start

        if current_team != team:
            # if the ball remains in possession with a team for a longer period of time then the window is added to
            # the record
            if duration >= min_duration_frames:
                record[window] = {
                    "window_num": window,
                    "team_in_possession": team,
                    "start_frame": window_start,
                    "end_frame": frame_num
                }

                window += 1

            # starts a new window and now focuses on the new team in possession
            window_start = frame_num
            team = current_team

        last_frame_num = frame_num

    # makes a final duration window for possession, so it is accounted
    if last_frame_num is None:
        return record

    final_duration = last_frame_num - window_start
    if final_duration >= min_duration_frames:
        record[window] = {
            "window_num": window,
            "team_in_possession": team,
            "start_frame": window_start,
            "end_frame": last_frame_num
        }

    return record


def load_image_positions(tracking_path, team_assignment_path) -> dict:
    """Ball and player positions in image space, with a per-frame pixel scale.

    Same shape as load_joined_positions, plus "scale": the median player box
    height in pixels, which stands in for PLAYER_HEIGHT_M and converts the
    possession radii into pixels under the current camera zoom.
    """
    records = defaultdict()

    tracking_records = load_records(tracking_path)
    team_records = load_records(team_assignment_path)

    for frame_num in sorted(tracking_records.keys() & team_records.keys()):
        ball_position = None
        player_positions = {}
        heights = []

        for detection in tracking_records[frame_num]["detections"]:
            x1, y1, x2, y2 = detection["bbox"]

            if detection["class"] == "Ball":
                ball_position = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            elif detection["class"] in ("Player", "Goalkeeper"):
                # the feet are where the player meets the pitch, so distances
                # between players are least distorted there
                player_positions[detection["track_id"]] = np.array([(x1 + x2) / 2, y2])
                heights.append(y2 - y1)

        records[frame_num] = {
            "ball_position": ball_position,
            "player_positions": player_positions,
            "player_teams": {p["track_id"]: p["team"] for p in team_records[frame_num]["players"]},
            "scale": float(np.median(heights)) if heights else None,
        }

    return records


def load_joined_positions(tracking_path, keypoint_output_path, team_assignment_path) -> dict:
    # return record
    records = defaultdict()

    # creating temp records to hold information to grab for the return record
    tracking_records = load_records(tracking_path)
    transformers = build_transformer_for_frame(keypoint_output_path)  # builds transformer to make vertices into real coordinates
    team_records = load_records(team_assignment_path)

    # grabs the common frames from all the different records
    common_frames = sorted(tracking_records.keys() & transformers.keys() & team_records.keys())

    # iterate over all the frames that are shared between the dicts
    for frame_num in common_frames:
        tracking_frame = tracking_records[frame_num]
        homography_frame = transformers[frame_num]
        team_frame = team_records[frame_num]

        # resets every iteration
        ball_position = None
        player_positions = {}

        # checks all detections for if it is a ball or player to seperate concerns
        for detection in tracking_frame["detections"]:
            if detection["class"] == "Ball":
                x1, y1, x2, y2 = detection["bbox"]
                ball_pixel_point = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]])  # shape (1, 2)

                ball_position = transformers[frame_num].transform_points(ball_pixel_point)[0]  # shape (2,) after indexing
            elif detection["class"] in ("Player", "Goalkeeper"):
                x1, y1, x2, y2 = detection["bbox"]
                player_pixel_point = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]])

                player_positions[detection["track_id"]] = transformers[frame_num].transform_points(player_pixel_point)[0]

        player_teams = {p["track_id"]: p["team"] for p in team_frame["players"]}

        records[frame_num] = {
            "ball_position": ball_position,
            "player_positions": player_positions,
            "player_teams": player_teams,
        }

    return records


#
def run_event_detection(tracking_path, keypoint_output_path, team_assignment_path, output_path, joined_positions=None):
    # image space, so possession is not limited to frames where the pitch
    # keypoints happened to resolve
    records = load_image_positions(tracking_path, team_assignment_path)

    # creates list for holding all possession frames
    per_frame_possession = []

    # iterates through all the frames in the records (the num of keys)
    for frame in sorted(records.keys()):
        record_frame = records[frame]
        scale = record_frame["scale"]

        if scale is None:
            possession_for_frame = {"track_id": None, "status": "No Ball Detected",
                                    "min_distance": None, "team": None}
        else:
            # convert the real-world radii into pixels at this frame's zoom
            possession_for_frame = detect_possession_for_frame(
                record_frame["ball_position"], record_frame["player_positions"],
                record_frame["player_teams"],
                r_possession=scale * POSSESSION_RADIUS_M / PLAYER_HEIGHT_M,
                r_duel=scale * DUEL_RADIUS_M / PLAYER_HEIGHT_M,
            )

        possession_for_frame["frame"] = frame
        per_frame_possession.append(possession_for_frame)  # appends the possession in the frame into a list

    # builds out the possession windows
    possession_windows = build_possession_windows(per_frame_possession)

    # writes it into a file
    with open(str(output_path), "w") as f:
        for window in possession_windows.values():
            f.write(json.dumps(window) + "\n")


# helper function which skims a jsonl file of records and returns a dict
def load_records(file_path) -> dict:
    records = defaultdict()

    with open(str(file_path), "r") as f:
        for line in f:
            clean_line = line.strip()

            if not clean_line:
                continue

            parsed = json.loads(clean_line)
            records[parsed["frame"]] = parsed

    return records
