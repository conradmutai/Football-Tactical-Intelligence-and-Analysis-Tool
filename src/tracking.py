import json

import cv2
import torch

from pathlib import Path

from ultralytics import YOLO

# creating a video path for testing
VIDEO_PATH = Path(__file__).resolve().parent.parent / "data" / "soccernet" / "england_epl" / "2014-2015" / "2015-02-21 - 18-00 Chelsea 1 - 1 Burnley" / "1_720p.mkv"

# creating a weight path for testing
WEIGHTS_PATH = Path(__file__).resolve().parent / "runs" / "detect" / "train-7" / "weights" / "best.pt"

# tracking output pathway
TRACKING_OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "tracking_output" / "2015-02-21_chelsea_burnley.jsonl"


def tracking(weights_path, video_path):
    model = YOLO(str(weights_path))
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)

    idx = 0

    # opening and writing in the tracking json
    with open(str(TRACKING_OUTPUT_PATH), "w") as f:
        # goes over each frame in the video capture
        while cap.isOpened():
            success, frame = cap.read()

            if success:
                # creates a tracker for players
                results = model.track(frame, persist=True, show=False, tracker="bytetrack.yaml")

                detections = []
                for box in results[0].boxes:
                    detections.append({
                        "track_id": int(box.id) if box.id is not None else None,  # the id for each object/person
                        "class": model.names[int(box.cls)],  # the class for each person
                        "bbox": [round(x, 1) for x in box.xyxy[0].tolist()],  # the bounding box of an object
                        "confidence": round(float(box.conf), 3),
                    })

                # creates a record for each frame
                record = {"frame": idx, "timestamp": idx / fps, "detections": detections}
                f.write(json.dumps(record) + "\n")

                idx += 1
            else:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    tracking(WEIGHTS_PATH, VIDEO_PATH)
