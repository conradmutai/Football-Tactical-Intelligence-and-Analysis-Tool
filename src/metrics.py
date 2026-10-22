import json
from collections import defaultdict
from typing import Dict, List

import numpy as np

from src.event_detection import euclidian_distance, load_joined_positions, load_records
from src.formation import load_json

FPS = 25

SPRINT_SPEED_THRESHOLD = 700  # units: pitch-coordinate units per second
# 48 units/frame at 25 fps is 12 m/s -- a shade above an elite sprint, so any
# larger step is homography jitter rather than a player. The old 300 (75 m/s)
# let 84% of impossible steps through and inflated distances several-fold.
MAX_PLAUSIBLE_STEP_DISTANCE = 48

# the pitch keypoints are noisy enough that projected positions jitter; a short
# median filter removes most of it without blunting real movement
TRAJECTORY_SMOOTHING_RADIUS = 5

# a burst has to last this long to count as a sprint. Without it, position noise
# flicks across the speed threshold repeatedly and reports double-figure sprints
# inside a ten-second window.
MIN_SPRINT_FRAMES = 12

PITCH_LENGTH = 12000
PITCH_WIDTH = 7000
ZONE_GRID_ROWS = 3  # along pitch WIDTH
ZONE_GRID_COLS = 3  # along pitch LENGTH


# builds every player's trajectory in a single pass over the window
def build_trajectory_index(joined_positions: dict, start_frame: int, end_frame: int,
                           smoothing_radius: int = TRAJECTORY_SMOOTHING_RADIUS) -> Dict[int, List[tuple]]:
    """Return {track_id: [(frame, position), ...]} for one window.

    This replaced a per-track_id scan that walked the whole window for each
    player and compared every id with `==` instead of a dict lookup. On a
    4,482-frame window with 980 track ids that was ~48M operations; both factors
    grow with clip length, so the old form was effectively quadratic and was the
    wall for long clips.
    """
    index = defaultdict(list)

    for frame_num in range(start_frame, end_frame + 1):
        frame_data = joined_positions.get(frame_num)

        if frame_data is None:
            continue

        for track_id, position in frame_data["player_positions"].items():
            index[track_id].append((frame_num, position))

    if smoothing_radius:
        for track_id, trajectory in index.items():
            index[track_id] = smooth_trajectory(trajectory, smoothing_radius)

    return index


def smooth_trajectory(trajectory: List[tuple], radius: int) -> List[tuple]:
    """Median-filter a trajectory's positions over a +/- radius frame window."""
    if len(trajectory) < 3:
        return trajectory

    frames = [frame_num for frame_num, _ in trajectory]
    positions = np.asarray([position for _, position in trajectory])

    smoothed = [
        np.median(positions[max(0, i - radius): i + radius + 1], axis=0)
        for i in range(len(positions))
    ]

    return list(zip(frames, smoothed))


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
        # the limit is per frame, so a step spanning several frames is allowed to
        # be correspondingly longer -- comparing a multi-frame step against the
        # single-frame limit silently discarded real movement
        if frame_num_diff > 5 or dist_between_points > MAX_PLAUSIBLE_STEP_DISTANCE * frame_num_diff:
            prev_positions = positions
            prev_frame_num = frame_num
            continue

        # sums the distances
        sum_dist += dist_between_points

        prev_positions = positions
        prev_frame_num = frame_num

    return sum_dist


# calculates the sprints speed as well as amount of sprints a player does
def compute_sprint_count(trajectory: List[tuple], fps: float, sprint_threshold: float,
                         min_sprint_frames: int = MIN_SPRINT_FRAMES) -> int:
    # initializes the sprinting boolean and count variable
    count = 0
    currently_sprinting = False
    sprint_start_frame = None

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
                currently_sprinting = True
                sprint_start_frame = frame_num
        else:
            # only counts once the burst proved it lasted long enough to be real
            if currently_sprinting and frame_num - sprint_start_frame >= min_sprint_frames:
                count += 1
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
        joined_positions: dict, start_frame: int, end_frame: int, track_ids: List[int], fps: float = FPS
) -> List[dict]:
    # stores a list of dicts for results
    res = []

    # one pass over the window gives every player's trajectory at once
    trajectories = build_trajectory_index(joined_positions, start_frame, end_frame)

    # loops over the players and checks their strategies, then gathers their stats
    for track_id in track_ids:
        trajectory = trajectories.get(track_id, [])

        # calculates the accuracy
        distance_covered = computing_distance_covered(trajectory)
        sprint_count = compute_sprint_count(trajectory, fps=fps, sprint_threshold=SPRINT_SPEED_THRESHOLD)
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
    possession_windows_path, tracking_path, keypoint_output_path, team_assignment_path, output_path,
    joined_positions=None, fps: float = FPS
):
    # loads in the possession windows from the json and joins frame per positions
    possession_windows = load_json(possession_windows_path)
    joined_per_frame_positions = joined_positions if joined_positions is not None else load_joined_positions(
        tracking_path, keypoint_output_path, team_assignment_path)

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
            track_ids,
            fps=fps
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