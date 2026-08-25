import json
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from src.event_detection import euclidian_distance, load_joined_positions

# pre declared formations for model to match to
FORMATION_TEMPLATES = {
    "4-4-2": [
        (2000, 2800), (2000, 4200),           # CBs
        (2500, 700), (2500, 6300),            # FBs
        (5000, 2800), (5000, 4200),           # CMs
        (5500, 700), (5500, 6300),            # wide mids
        (8500, 2800), (8500, 4200),           # STs
    ],
    "4-3-3": [
        (2000, 2800), (2000, 4200),           # CBs
        (2500, 700), (2500, 6300),            # FBs
        (4500, 3500),                          # DM
        (5500, 2400), (5500, 4600),           # CMs
        (8000, 700), (8000, 6300),            # wingers
        (9000, 3500),                          # ST
    ],
    "4-2-3-1": [
        (2000, 2800), (2000, 4200),           # CBs
        (2500, 700), (2500, 6300),            # FBs
        (4500, 2800), (4500, 4200),           # DMs
        (6800, 1400), (6800, 5600),           # wide AMs
        (7000, 3500),                          # CAM
        (9000, 3500),                          # ST
    ],
    "4-1-4-1": [
        (2000, 2800), (2000, 4200),           # CBs
        (2500, 700), (2500, 6300),            # FBs
        (4000, 3500),                          # DM
        (5800, 700), (5800, 2400), (5800, 4600), (5800, 6300),  # midfield 4
        (9000, 3500),                          # ST
    ],
    "3-5-2": [
        (2000, 1750), (2000, 3500), (2000, 5250),  # CBs
        (4000, 700), (4000, 6300),                  # WBs
        (5500, 1750), (5500, 3500), (5500, 5250),  # CMs
        (8500, 2800), (8500, 4200),                 # STs
    ],
    "3-4-3": [
        (2000, 1750), (2000, 3500), (2000, 5250),  # CBs
        (4500, 700), (4500, 6300),                  # WBs
        (5500, 2800), (5500, 4200),                 # CMs
        (8000, 1400), (8000, 5600),                 # wingers
        (8800, 3500),                                # ST
    ],
    "5-3-2": [
        (1800, 1750), (1800, 3500), (1800, 5250),  # CBs
        (3500, 700), (3500, 6300),                  # WBs
        (5500, 1750), (5500, 3500), (5500, 5250),  # CMs
        (8500, 2800), (8500, 4200),                 # STs
    ],
    "4-5-1": [
        (2000, 2800), (2000, 4200),           # CBs
        (2500, 700), (2500, 6300),            # FBs
        (5500, 700), (5500, 2100), (5500, 3500), (5500, 4900), (5500, 6300),  # midfield 5
        (9000, 3500),                          # ST
    ],
    "4-4-1-1": [
        (2000, 2800), (2000, 4200),           # CBs
        (2500, 700), (2500, 6300),            # FBs
        (5500, 700), (5500, 2400), (5500, 4600), (5500, 6300),  # midfield 4
        (7500, 3500),                          # 2nd striker
        (9000, 3500),                          # ST
    ],
    "3-4-2-1": [
        (2000, 1750), (2000, 3500), (2000, 5250),  # CBs
        (4500, 700), (4500, 6300),                  # WBs
        (5500, 2800), (5500, 4200),                 # CMs
        (7500, 2100), (7500, 4900),                 # AMs
        (9000, 3500),                                # ST
    ],
    "4-1-2-1-2": [  # diamond
        (2000, 2800), (2000, 4200),           # CBs
        (2500, 700), (2500, 6300),            # FBs
        (4000, 3500),                          # DM
        (5800, 2400), (5800, 4600),           # CMs
        (7200, 3500),                          # CAM
        (8800, 2800), (8800, 4200),           # STs
    ],
}


# compares positions of players to positions in formation templates
def match_positions_to_template(observed_positions: List[np.ndarray], template_positions: List[Tuple[float, float]]) -> float:
    # initializing iteration loop and cost matrix
    cost_matrix = np.zeros((len(observed_positions), len(template_positions)))
    i = j = 0

    # loops over the observed positions
    for obs_pos in observed_positions:
        for template_pos in template_positions:  # compares the observed position to all the template positions
            # calculates euclidian distance and then assigns it to point in cost matrix
            dist = euclidian_distance(obs_pos, np.asarray(template_pos))
            cost_matrix[i][j] = dist

            j += 1

        j = 0
        i += 1

    # scipy's implementation of the Hungarian / Munkres algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Sum of optimal pairings
    return cost_matrix[row_ind, col_ind].sum()


# classifies the formations
def classify_formation(observed_positions: List[np.ndarray]) -> list[str, float]:
    # initializing a minimum distance
    min_sum_dist = float("inf")

    # loops over every formation in the templates and then calculates the sum of distances and then compares to get the min
    for formation_name, formation in FORMATION_TEMPLATES.items():
        sum_dist = match_positions_to_template(observed_positions, formation)
        if sum_dist < min_sum_dist:
            min_sum_dist = sum_dist
            best_formation_name = formation_name

    # returns a list containing the formation and the min dist
    return [best_formation_name, min_sum_dist]


# calculates the average positions in each window
def average_positions_for_window(joined_positions: dict, start_frame: int, end_frame: int, team: int) -> list:
    # creates a list to grab valid positions by player
    positions_by_player = defaultdict(list)

    # iterates through all frames in a window to see if the frame consists of valid player positions and teams
    for frame_num in range(start_frame, end_frame + 1):
        frame_data = joined_positions[frame_num]

        for track_id, positions in frame_data["player_positions"].items():
            if frame_data["player_teams"].get(track_id) == team:
                positions_by_player[track_id].append(positions)

    # dictionary to return
    averaged_positions = []

    # iterates through positions by player
    for track_id, positions_array in positions_by_player.items():
        positions_array = np.asarray(positions_array)  # converts the array into a np array
        avg_pos = np.mean(positions_array, axis=0)
        averaged_positions.append(avg_pos)

    return averaged_positions


# classifies the formation for each window
def classify_windows(possession_windows_path, tracking_path, keypoint_output_path, team_assignment_path) -> List[dict]:
    # loads the json file and the joined positions
    possession_windows = load_json(possession_windows_path)
    joined_per_frame_possession = load_joined_positions(tracking_path, keypoint_output_path, team_assignment_path)

    # result array for returning the result
    result = []

    # iterates through the different possession windows
    for window in possession_windows.values():
        # gathers the attacking team and possession teams
        possessing_team = window["team_in_possession"]
        defending_team = 1 - possessing_team

        # gets the offensive and defensive positions for the teams
        offensive_positions = average_positions_for_window(
            joined_per_frame_possession, window["start_frame"], window["end_frame"], possessing_team)
        defensive_positions = average_positions_for_window(
            joined_per_frame_possession, window["start_frame"], window["end_frame"], defending_team)

        # classifies the formations for each of the teams
        offensive_classified_possession = classify_formation(offensive_positions)
        defensive_classified_possession = classify_formation(defensive_positions)

        # creates a dict storing all this information
        record = {
            "window": window["window_num"],
            "attacking_team": possessing_team,
            "offensive_formation": offensive_classified_possession,
            "defensive_team": defending_team,
            "defensive_formation": defensive_classified_possession
        }

        result.append(record)

    return result


# helper function which skims a jsonl file of records and returns a dict
def load_json(file_path) -> dict:
    records = defaultdict()

    with open(str(file_path), "r") as f:
        for line in f:
            clean_line = line.strip()

            if not clean_line:
                continue

            parsed = json.loads(clean_line)
            records[parsed["window_num"]] = json.loads(clean_line)

    return records
