import json
import cv2
import numpy as np
import supervision as sv

from ultralytics import YOLO


def keypoint_detection(video_path, weight_path, keypoint_output_path, max_frames=None):
    # model and video preparation
    model = YOLO(str(weight_path))
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_number = 1

    # Opening the keypoint output json and writing data in it
    with open(str(keypoint_output_path), "w") as f:
        while cap.isOpened():
            success, frame = cap.read()

            if not success:
                break

            result = model.predict(frame)[0]  # runs inference on the frame
            key_points = sv.KeyPoints.from_ultralytics(result)  # gets the key points from the inferred frame

            if key_points.keypoint_confidence is None:
                # no detections this frame — write an empty record and move on
                record = {
                    "frame": frame_number,
                    "timestamp": frame_number / fps,
                    "indices": [],
                    "filtered_xy": [],
                    "filtered_confidence": []
                }
                f.write(json.dumps(record) + "\n")
                frame_number += 1
                continue

            indices = np.where(key_points.keypoint_confidence[0] > 0.3)[0]  # helps with masking key points with confidence below 0.3
            filtered_xy = key_points.xy[0][indices]  # coordinates, in the same order as `indices`
            filtered_confidence = key_points.keypoint_confidence[0][indices]

            # saves all the information into a JSON line file
            record = {
                "frame": frame_number,
                "timestamp": frame_number / fps,
                "indices": indices.tolist(),
                "filtered_xy": filtered_xy.tolist(),
                "filtered_confidence": filtered_confidence.tolist()
            }
            f.write(json.dumps(record) + "\n")

            frame_number += 1

            if max_frames is not None and frame_number > max_frames:
                break

