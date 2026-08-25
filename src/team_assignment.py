import json
import random
from collections import defaultdict
from typing import List
from sklearn.cluster import KMeans

import cv2
import numpy as np


# Handles the identification of teams to enable proper formation analysis
class TeamAssigner:
    def __init__(self, n_clusters: int = 2, random_state: int = 42):  # parameters are fed into KMeans to make cluster model
        self.cluster_model = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.is_fitted = False

    # extracts the hue and saturation from a frame to allow for proper team identification
    def extract_frame(self, crops: List[np.ndarray]) -> np.ndarray:
        hs_values = []

        # pulls a crop from list of crops
        for crop in crops:
            height = crop.shape[0]

            # cuts of the head and legs/shorts isolating the jersey
            start_row = int(height * 0.2)
            end_row = int(height*0.6)
            torso_region = crop[start_row:end_row, :, :]

            # changes color scheme to HSV (Hue, Saturation, Value) and extracts Hue and Saturation
            color_change = cv2.cvtColor(torso_region, cv2.COLOR_BGR2HSV)
            hsv = np.mean(color_change, axis=(0, 1))
            mean_h_s = hsv[:2]

            hs_values.append(mean_h_s)

        # converts Python List to NumPy array
        hs_values = np.asarray(hs_values)

        return hs_values

    # fits the sample crops to the cluster model
    def fit(self, sample_crops: List[np.ndarray]) -> None:
        result = self.extract_frame(sample_crops)   # crops and color filters player bbox
        self.cluster_model.fit(result)
        self.is_fitted = True  # changes fitted state to true to indicate it has been carried out

    # predicts the team labels of players utilizing fitted color-model
    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        # barriers to prevent any errors or crashes with model running and also to help with identifying errors
        if not self.is_fitted:
            raise RuntimeError(".predict() was ran before .fit()")

        if len(crops) == 0:
            return (np.
                    array([]))

        # recreates hue and saturation values and then predicts with said values
        hs_values = self.extract_frame(crops)
        return self.cluster_model.predict(hs_values)


# Returns a cropped player image
def extract_crop(frame: np.ndarray, bbox: dict) -> np.ndarray:
    x1, y1, x2, y2 = np.array(bbox).astype(int)  # extracts the xyxy details from bbox and converts it to a numpy array then int
    frame = frame[y1:y2, x1:x2]  # modifies the frame to fit the constraints of the bbox
    return frame


# verifies if a character is a referee and if so it excludes them from classification
def is_referee(record: dict) -> bool:
    person_class = record["class"]

    if person_class != "Referee":
        return False

    return True


def load_tracking_records(tracking_path: str):
    # creates a dictionary for the records
    records = defaultdict()

    # opens file for video id and goes line by line adding clean_lines to the record
    with open(tracking_path, "r") as file:
        for line in file:
            clean_line = line.strip()

            if not clean_line:
                continue

            parsed = json.loads(clean_line)
            records[parsed["frame"]] = (json.loads(clean_line))  # adds clean line to records

    return records


# creates a sample of crops which will then be fed in for feeding
def sample_crops_for_fitting(cap, tracking_records, n_frames: int = 50):
    # gets the total amount of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # grabs the minimum amount of frames between the two and then creates indices on what to select
    n_frames = min(n_frames, total_frames)
    sampled_indices = random.sample(range(total_frames), n_frames)

    # empty list for the list of crops
    crops = []

    # iterates over the frame indexes in the sample indices and then selects the crops for fitting
    for frame_idx in sampled_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = cap.read()
        if not success:
            continue

        record = tracking_records.get(frame_idx + 1)  # tracking.py's frame numbers start at 1
        if record is None:
            continue

        # checks for if there is a detection
        for detection in record["detections"]:
            if is_referee(detection):
                continue
            crop = extract_crop(frame, detection["bbox"])
            crops.append(crop)

    return crops


def assign_teams_for_video(video_path, tracking_path, output_path, max_frames=None):
    # grabs tracking record from the video id
    records = load_tracking_records(tracking_path)

    # makes the video into a cv2 capture and gathers total frames and fps
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fitting_sample = sample_crops_for_fitting(cap=cap, tracking_records=records)
    team_assigner = TeamAssigner()
    team_assigner.fit(fitting_sample)

    # sets the cap back to frame 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_idx = 1

    with open(str(output_path), "w") as f:
        while cap.isOpened():
            # grabs frame from capture and grabs a record for a certain frame
            success, frame = cap.read()

            if not success:
                break

            # gathers the frame record utilizing the frame index
            frame_record = records[frame_idx]

            # gathers the crops
            crops = []
            track_ids = []

            # loops over all the detections in a frame record
            for detection in frame_record["detections"]:
                if is_referee(detection):
                    continue
                crop = extract_crop(frame, detection["bbox"])
                crops.append(crop)
                track_ids.append(detection["track_id"])

            # gathers hue and saturation values from frame
            team_labels = team_assigner.predict(crops)

            # creates a list of players and has a tracking id for each player and labels their team
            players = [
                {"track_id": tid, "team": int(label)}
                for tid, label in zip(track_ids, team_labels)
            ]

            # record is then created and information is then stored
            record = {
                "frame": frame_idx,
                "timestamp": frame_idx / fps,
                "players": players
            }

            f.write(json.dumps(record) + "\n")

            frame_idx += 1

            if max_frames is not None and frame_idx > max_frames:
                break

    # clears the frames
    cap.release()

