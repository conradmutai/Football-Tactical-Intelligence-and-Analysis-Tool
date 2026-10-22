import json
from pathlib import Path

import cv2
import numpy as np

from sports.configs.soccer import SoccerPitchConfiguration

PITCH_CONFIG = SoccerPitchConfiguration()

# The pitch-keypoint model emits all 32 keypoints for every frame whether or not
# the landmark is visible, and the low-confidence ones land on empty grass. Fitting
# a homography to those gave a median reprojection error of 3.3 m, with 29% of
# frames worse than 10 m -- which is what made pitch-space distances come out at
# 80+ m/s. Keeping only confident keypoints and validating the fit takes the
# median error to 0.2 m at the same ~52% frame coverage.
MIN_KEYPOINT_CONFIDENCE = 0.7
MIN_KEYPOINTS = 6          # 4 would fit any homography exactly and prove nothing
RANSAC_REPROJ_THRESHOLD = 5.0
MAX_REPROJECTION_ERROR = 300.0  # pitch units; 100 units = 1 m

# A fit can match its own landmarks perfectly and still be useless if those
# landmarks sit in one small part of the image -- it is then ill-conditioned and
# extrapolates nonsense out to where the players are. These check the resulting
# transform rather than the fit: the keypoints must span real image area, and the
# video frame must map to a sane convex patch of pitch.
MIN_KEYPOINT_IMAGE_SPREAD = 0.05  # fraction of frame area the keypoints must cover
MAX_PITCH_COORDINATE = 30000.0    # a frame corner this far off-pitch means a broken fit
FRAME_CORNERS = np.array([[0, 0], [1280, 0], [1280, 720], [0, 720]], dtype=np.float32)

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

        self.m, self.inlier_mask = cv2.findHomography(
            source, target, cv2.RANSAC, ransacReprojThreshold=RANSAC_REPROJ_THRESHOLD
        )  # creates the homography matrix
        if self.m is None:
            raise ValueError("Homography matrix calculation failed")

        # how far the fit misses the pitch landmarks it was built from; the
        # caller uses this to throw out frames the geometry does not support
        projected = self.transform_points(source)
        self.reprojection_error = float(np.median(np.linalg.norm(projected - target, axis=1)))

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


def _is_well_conditioned(source, transformer, frame_shape=(720, 1280)):
    """Reject fits built from a cramped cluster of points, or that fold the pitch."""
    height, width = frame_shape

    if cv2.contourArea(cv2.convexHull(source)) < MIN_KEYPOINT_IMAGE_SPREAD * height * width:
        return False

    corners = transformer.transform_points(FRAME_CORNERS)

    # a valid perspective map keeps the frame a convex quadrilateral
    if len(cv2.convexHull(corners.astype(np.float32))) != 4:
        return False

    return bool(np.abs(corners).max() <= MAX_PITCH_COORDINATE)


def build_transformer_for_frame(keypoint_output_path,
                                min_confidence=MIN_KEYPOINT_CONFIDENCE,
                                min_keypoints=MIN_KEYPOINTS,
                                max_error=MAX_REPROJECTION_ERROR):
    """Build one validated ViewTransformer per frame.

    Frames whose keypoints cannot support a trustworthy homography are left out
    entirely rather than contributing a wrong one -- downstream stages already
    skip missing frames, and a missing radar is better than a wrong one.
    """
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

            # keep only keypoints the model is actually confident about
            confidences = record.get("filtered_confidence") or []
            keep = [
                (index, point)
                for index, point, confidence in zip(record["indices"], record["filtered_xy"], confidences)
                if confidence >= min_confidence
            ]

            if len(keep) < min_keypoints:
                continue  # not enough points for a reliable homography

            # gets the source from the filtered_xy and the target coordinates from the pitch_config
            source = np.array([point for _, point in keep], dtype=np.float32)
            target = np.array(PITCH_CONFIG.vertices, dtype=np.float32)[[index for index, _ in keep]]

            try:
                transformer = ViewTransformer(source, target)
            except ValueError:
                continue

            if transformer.reprojection_error > max_error:
                continue  # the fit does not agree with the landmarks it came from

            if not _is_well_conditioned(source, transformer):
                continue

            transformers[frame] = transformer

    return transformers
