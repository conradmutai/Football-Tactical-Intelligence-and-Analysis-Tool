import cv2
import numpy as np


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
