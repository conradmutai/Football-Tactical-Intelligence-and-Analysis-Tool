import cv2
from ultralytics import YOLO
from pathlib import Path

# creating a path to the config file to use to initialize the model
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"

# creating a video path
VIDEO_PATH = Path(__file__).resolve().parent.parent / "data" / "soccernet" / "england_epl" / "2014-2015" / "2015-02-21 - 18-00 Chelsea 1 - 1 Burnley" / "1_720p.mkv"

# creating a weight path
weights_path = Path(__file__).resolve().parent / "runs" / "detect" / "train-7" / "weights" / "best.pt"

# # loading YOLO
# model = YOLO("yolov9s.pt")   # YOLO v9 tiny (pre-trained model)
#
# # training the model
# if __name__ == '__main__':
#     results = model.train(
#         data=str(CONFIG_PATH),
#         epochs=100,
#         imgsz=640,
#         batch=16,
#         optimizer="AdamW",
#         device="mps",
#         name="train-7")


def train():
    # loading custom weights over random weights for applying the model
    model = YOLO(str(weights_path))  # loading the best weights into the model for training

    video_path = str(VIDEO_PATH)
    cap = cv2.VideoCapture(video_path)  # capture of the video

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            results = model.track(frame, persist=True, show=False, tracker='botsort.yaml')
            annotated_frame = results[0].plot()  # plots an annotated frame for the tracker
            cv2.imshow("YOLOv9 Tracking", annotated_frame)

            # if q is held then it breaks
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    train()
