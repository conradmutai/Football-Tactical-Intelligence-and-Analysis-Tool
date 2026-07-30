from ultralytics import YOLO
from pathlib import Path

# creating a path to the config file to use to initialize the model
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"


def detect():
    # loading YOLO
    model = YOLO("yolov9s.pt")  # YOLO v9 tiny (pre-trained model)

    # training the model
    model.train(
        data=str(CONFIG_PATH),
        epochs=100,
        imgsz=640,
        batch=16,
        optimizer="AdamW",
        device="mps",
        name="train-7")


