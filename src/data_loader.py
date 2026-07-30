import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()  # reads .env and loads variables into the environment
api_key = os.getenv("ROBOFLOW_API_KEY")


def dataset():
    rf = Roboflow(api_key=api_key)  # fetches data using my api key
    project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
    version = project.version(1)
    dataset = version.download("yolov9")

    # Moving the data into a data folder
    dest = Path("data/dataset")
    if dest.exists():
        shutil.rmtree(dest)
    shutil.move(dataset.location, dest)


def dataset_pose():
    rf = Roboflow(api_key=api_key)  # fetches data using my api key
    project = rf.workspace("roboflow-jvuqo").project("football-field-detection-f07vi")  # gets data from roboflow to train model
    version = project.version(14)
    dataset = version.download("yolov8")

    # Moving the data into a data folder
    dest = Path("data/dataset_pose")
    if dest.exists():
        shutil.rmtree(dest)
    shutil.move(dataset.location, dest)
