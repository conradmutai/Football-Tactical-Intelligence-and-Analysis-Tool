import json
from pathlib import Path

import cv2
import numpy as np

from sports.configs.soccer import SoccerPitchConfiguration

PITCH_CONFIG = SoccerPitchConfiguration()

# Tracking Path for testing
KEYPOINT_OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "keypoint_output" / "2015-02-21_chelsea_burnley.jsonl"


class ViewTransformer:
    # source (xy) and target (vertices) are numpy arrays
    def __init__(self, source, target):
        if source.shape != target.shape:
            raise ValueError("Source and Target must have the same shape")
        if source.shape[1] != 2:
            raise ValueError("Source and target have to be 2D")

        # transforms the source and target into np floats
        source = source.astype(np.float32)
        target = target.astype(np.float32)

        self.m, _ = cv2.findHomography(source, target, cv2.RANSAC)  # creates the homography matrix
        if self.m is None:
            raise ValueError("Homography matrix calculation failed")

    # points what were retrieved from the JSONL xy
    def transform_points(self, points):
        if points.size == 0:
            return points

        if points.shape[1] != 2:
            raise ValueError("Points must be 2D")

        points = points.reshape(-1, 1, 2).astype(
            np.float32)  # reshapes the points into an applicable form for perspective transform
        points = cv2.perspectiveTransform(points, self.m)  # applies perspective transform to points
        return points.reshape(-1, 2).astype(np.float32)


def build_transformer_for_frame(keypoint_output_path):
    transformers = {}

    # reads through each line of the keypoint path
    with open(str(keypoint_output_path), "r") as f:
        for line in f:
            clean_line = line.strip()

            # skips over if there is an existing issue
            if not clean_line:
                continue

            record = json.loads(clean_line)
            frame = record["frame"]
            indices = record["indices"]
            filtered_xy = record["filtered_xy"]

            if len(indices) < 4:
                continue  # not enough points for a reliable homography

            # gets the source from the filtered_xy and the target coordinates from the pitch_config
            source = np.array(filtered_xy, dtype=np.float32)
            target = np.array(PITCH_CONFIG.vertices, dtype=np.float32)[indices]

            transformers[frame] = ViewTransformer(source, target)

    return transformers
