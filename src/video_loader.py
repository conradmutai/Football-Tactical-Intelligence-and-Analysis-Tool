import os

from dotenv import load_dotenv
from SoccerNet.Downloader import SoccerNetDownloader
import urllib.request

load_dotenv()  # reads .env and loads variables into the environment
soccer_net_password = os.getenv("SOCCER_NET_PASSWORD")

mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory="data/soccernet")
mySoccerNetDownloader.password = soccer_net_password
mySoccerNetDownloader.downloadGames(files=["1_720p.mkv", "2_720p.mkv"], split=["train","valid","test","challenge"])
mySoccerNetDownloader.downloadGames(files=["1_224p.mkv", "2_224p.mkv"], split=["train","valid","test","challenge"])
