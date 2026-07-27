import cv2
from ultralytics import YOLO
from pathlib import Path

# creating a path to the config file to use to initialize the model
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"

# # loading YOLO
# model = YOLO("yolov9m.pt")   # YOLO v9 medium
#
# # training the model
# if __name__ == '__main__':
#     results = model.train(data=str(CONFIG_PATH))

# loading custom weights over random weights for applying the model
# model = YOLO('runs/detect/train/weights/best.pt')


def train():
    model = YOLO('runs/detect/train-4/weights/best.pt')  # loading the best weights into the model for training

    video_path = r""
    cap = cv2.VideoCapture(video_path)  # capture of the video

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            results = model.track(frame, persist=True, show=True, tracker='botsort.yaml')
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
