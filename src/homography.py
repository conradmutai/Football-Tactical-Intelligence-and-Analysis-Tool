import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()  # reads .env and loads variables into the environment
api_key = os.getenv("ROBOFLOW_API_KEY")


