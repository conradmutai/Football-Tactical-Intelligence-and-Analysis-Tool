from typing import List
from sklearn.cluster import KMeans

import cv2
import numpy as np


# Handles the identification of teams to enable proper formation analysis
class TeamClassifier:
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
            return np.ndarray([])

        # recreates hue and saturation values and then predicts with said values
        hs_values = self.extract_frame(crops)
        return self.cluster_model.predict(hs_values)


def extract_crop(frame: np.ndarray, bbox: dict) -> np.ndarray:
    ...


def is_referee(record: dict) -> bool:
    ...


def load_tracking_records(video_id: str):
    ...


def sample_crops_for_fitting(frames_source, tracking_records, n_frames: int = 50):
    ...


def assign_teams_for_video(video_path, video_id):
    ...