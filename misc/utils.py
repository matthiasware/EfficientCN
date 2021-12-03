import time
import datetime
from pathlib import Path


def get_sting_timestamp():
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')


def mkdir_directories(dirs, parents, exist_ok):
    for director in dirs:
        Path(director).mkdir(parents=parents, exist_ok=exist_ok)
