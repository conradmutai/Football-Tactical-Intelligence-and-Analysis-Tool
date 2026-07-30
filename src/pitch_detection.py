from ultralytics import YOLO
from pathlib import Path

# Data.yaml file path
CONFIG_FILE = Path(__file__).parent.parent / "data" / "dataset_pose" / "data.yaml"


def pitch_detect():
    model = YOLO("yolov8s-pose.pt")

    model.train(
        data=str(CONFIG_FILE),
        epochs=200,
        imgsz=640,
        batch=16,
        mosaic=0.0,
        optimizer="AdamW",
        device="mps",  # mac metal performance shader
        name="train-pose",
    )


if __name__ == '__main__':
    pitch_detect()
