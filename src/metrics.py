import json
from collections import defaultdict
from typing import Dict, List

import numpy as np

from src.event_detection import euclidian_distance, load_joined_positions, load_records
from src.formation import load_json

FPS = 25

SPRINT_SPEED_THRESHOLD = 700  # units: pitch-coordinate units per second
MAX_PLAUSIBLE_STEP_DISTANCE = 300

PITCH_LENGTH = 12000
PITCH_WIDTH = 7000
ZONE_GRID_ROWS = 3  # along pitch WIDTH
ZONE_GRID_COLS = 3  # along pitch LENGTH


# determines the players overall trajectory
def get_player_trajectory(joined_positions: dict, start_frame: int, end_frame: int, track_id: int) -> List[tuple]:
    # list for storing the tuples after each loop
    result = []

    # loops over the frames in the window
    for frame_num in range(start_frame, end_frame + 1):
        frame_data = joined_positions.get(frame_num)

        if frame_data is None:
            continue

        # goes over all the frames track ids (players) and positions
        for f_track_id, positions in frame_data["player_positions"].items():
            if f_track_id == track_id:  # if the player track id is equal to the player we are looking for then we append result
                result.append((frame_num, positions))

    # returns the list of track ids and positions
    return result


# computes the distance covered
def computing_distance_covered(trajectory: List[tuple]) -> float:
    # total distance a player covers in a time period and a previous posiitons
    sum_dist = 0.0
    prev_positions = None
    prev_frame_num = 0

    # loops over the positions in the player trajectory
    for frame_num, positions in trajectory:
        if prev_positions is None:  # checks if no previous positions exists and if not we assign it to the curr and skip
            prev_positions = positions
            prev_frame_num = frame_num
            continue

        # calculates the euclidian distance between each of the two positions
        dist_between_points = euclidian_distance(positions, prev_positions)

        frame_num_diff = frame_num - prev_frame_num
        if frame_num_diff > 5 or dist_between_points > MAX_PLAUSIBLE_STEP_DISTANCE:
            prev_positions = positions
            prev_frame_num = frame_num
            continue

        # sums the distances
        sum_dist += dist_between_points

        prev_positions = positions
        prev_frame_num = frame_num

    return sum_dist


# calculates the sprints speed as well as amount of sprints a player does
def compute_sprint_count(trajectory: List[tuple], fps: float, sprint_threshold: float) -> int:
    # initializes the sprinting boolean and count variable
    count = 0
    currently_sprinting = False

    # creates a list of frame numbers, and frame numbers matching to position for accessing the variables ahead
    frame_nums = {frame_num for frame_num, _ in trajectory}
    frame_num_to_pos = {frame_num: positions for frame_num, positions in trajectory}

    # loops over the frame_num and positions in the trajectory
    for frame_num, positions in trajectory:
        # looks 4 frames ahead
        frame_num_ahead = frame_num + 4

        # checks if the frame_num is in the list of frame numebrs
        if frame_num_ahead not in frame_nums:
            continue

        # gets the positions ahead utilizing the frame number ahead of the current
        positions_ahead = frame_num_to_pos[frame_num_ahead]

        # calculates the speed utilizing euclidian distance and time from elapsed and fps
        distance = euclidian_distance(positions_ahead, positions)
        time = (frame_num_ahead - frame_num) / fps
        speed = float(distance / time)

        # if speed is more than the current threshold and the player is not currently sprinting
        # then the count is incremented
        if speed > sprint_threshold:
            if not currently_sprinting:
                count += 1  # this is the START of a new sprint
                currently_sprinting = True
        else:
            currently_sprinting = False  # speed dropped — sprint has ended

    return count


# gets the zone that _ player is in
def get_zone(position: np.ndarray, pitch_length: float, pitch_width: float) -> tuple:
    # gets the percentage of the way through the length/width of the pitch a player is
    fraction_x = np.clip(position[0] / pitch_length, 0, 1)
    fraction_y = np.clip(position[1] / pitch_width, 0, 1)

    # gets the zone said player is in
    col = min(int(fraction_x * ZONE_GRID_COLS), ZONE_GRID_COLS - 1)
    row = min(int(fraction_y * ZONE_GRID_ROWS), ZONE_GRID_ROWS - 1)

    return row, col


# computes the zone occupancy
def compute_zone_occupancy(trajectory: List[tuple], pitch_length: float, pitch_width: float) -> Dict[tuple, int]:
    # tallying the amount of zones occupied
    tally = defaultdict(int)

    # goes over the frames and trajectory
    for frame_num, position in trajectory:
        zone = get_zone(position, pitch_length, pitch_width)
        tally[zone] += 1

    # returns the total tally
    return tally


# tracks a players metrics
def compute_metrics_for_window(
        joined_positions: dict, start_frame: int, end_frame: int, track_ids: List[int]
) -> List[dict]:
    # stores a list of dicts for results
    res = []

    # loops over the players and checks their strategies, then gathers their stats
    for track_id in track_ids:
        trajectory = get_player_trajectory(joined_positions, start_frame, end_frame, track_id)

        # calculates the accuracy
        distance_covered = computing_distance_covered(trajectory)
        sprint_count = compute_sprint_count(trajectory, fps=FPS, sprint_threshold=SPRINT_SPEED_THRESHOLD)
        zone_occupancy = compute_zone_occupancy(trajectory, pitch_length=PITCH_LENGTH, pitch_width=PITCH_WIDTH)

        # stores the information in a dict
        record = {
            "track_id": track_id,
            "distance_covered": float(distance_covered),
            "sprint_count": sprint_count,
            "zone_occupancy": {f"{row}_{col}": count for (row, col), count in zone_occupancy.items()}
        }

        # stores the dict into a list
        res.append(record)

    return res


def compute_metrics(
    possession_windows_path, tracking_path, keypoint_output_path, team_assignment_path, output_path
):
    # loads in the possession windows from the json and joins frame per positions
    possession_windows = load_json(possession_windows_path)
    joined_per_frame_positions = load_joined_positions(tracking_path, keypoint_output_path, team_assignment_path)

    # creates a list to store on json
    res = []

    # loops over the windows in the possession windows and gets statistics for player per possession window
    for window in possession_windows.values():
        # gathers the track ids (players) in the window
        track_ids = get_track_ids_in_window(
            joined_per_frame_positions,
            window["start_frame"],
            window["end_frame"]
        )

        # calls method to grab metrics for window called
        metrics_for_window = compute_metrics_for_window(
            joined_per_frame_positions,
            window["start_frame"],
            window["end_frame"],
            track_ids
        )

        # stores information in  adict
        record = {
            "window": window["window_num"],
            "window_metrics": metrics_for_window,
        }

        res.append(record)

    # writes the information into a jsonl file
    with open(str(output_path), "w") as f:
        for record in res:
            f.write(json.dumps(record) + "\n")

    return res


# helper function
def get_track_ids_in_window(joined_positions: dict, start_frame: int, end_frame: int) -> List[int]:
    track_ids = set()

    for frame_num in range(start_frame, end_frame + 1):
        frame_data = joined_positions.get(frame_num)
        if frame_data is None:
            continue

        track_ids.update(frame_data["player_positions"].keys())

    return list(track_ids)