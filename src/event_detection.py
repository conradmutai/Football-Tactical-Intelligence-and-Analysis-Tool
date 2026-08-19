import json
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import numpy as np

from src.homography import build_transformer_for_frame


# euclidian distance calculation
def eculidean_distance(point_a: np.ndarray, point_b: np.ndarray) -> float:
    x_diff = point_b[0] - point_a[0]
    y_diff = point_b[1] - point_a[1]
    dist = np.sqrt(np.square(y_diff) - np.square(x_diff))
    return dist


# finds the nearest player to the ball
def find_nearest_player(ball_position: np.ndarray, player_position: dict) -> Optional[List]:
    # initializing the return
    nearest_id = None
    min_dist = float('inf')

    # iterates through all player positions
    for track_id, player_pos in player_position.items():
        dist = eculidean_distance(ball_position, player_pos)

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
    if r_duel < min_dist < r_possession:
        status = "ball in transit"
    elif min_dist < r_duel:  # ball is in duel radius
        status = "in-duel"
    else:  # ball is far from anyone
        status = "no-one in possession"

    # creates a dict
    record = {
        "track_id": nearest_id,
        "status": status,
        "min_distance": min_dist,
        "team": player_teams.get(nearest_id)
    }

    return record


def build_possession_windows(per_frame_possession: List[Optional[dict]], min_durations: int = ...):
    for frame_possession in per_frame_possession:
        ...


def load_joined_positions(tracking_path, keypoint_output_path, team_assignment_path) -> dict:
    # return record
    records = defaultdict()

    # creating temp records to hold information to grab for the return record
    tracking_records = load_records(tracking_path)
    transformers = build_transformer_for_frame(keypoint_output_path)  # builds transformer to make vertices into real coordinates
    team_records = load_records(team_assignment_path)

    # grabs the common frames from all the different records
    common_frames = tracking_records.keys() & transformers.keys() & team_records.keys()

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
def run_event_detection(tracking_path, keypoint_output_path, team_assignment_path, output_path):
    records = load_joined_positions(tracking_path, keypoint_output_path, team_assignment_path)

    per_frame_possession = []

    for frame in records.keys():
        record_frame = records[frame]

        possession_for_frame = detect_possession_for_frame(record_frame["ball_position"], record_frame[frame], record_frame["player_teams"], r_possession=1.5, r_duel=2.5)
        per_frame_possession.append(possession_for_frame)

    possession_windows = build_possession_windows(per_frame_possession)

    frame_num = 1
    with open(str(output_path), "w") as f:
        for window in possession_windows:
            record = {
                "frame": frame_num,
                "possession_window": window
            }

            f.write(json.dumps(record) + "\n")

            frame_num += 1


# helper function which skims a jsonl file of records and returns a dict
def load_records(file_path) -> dict:
    records = defaultdict()

    with open(str(file_path), "r") as f:
        for line in f:
            clean_line = line.strip()

            if not clean_line:
                continue

            parsed = json.loads(clean_line)
            records[parsed["frame"]] = json.loads(clean_line)

    return records
