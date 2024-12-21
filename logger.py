import os
import shutil
from datetime import datetime


def log_directory_contents(log_directory):
    current_directory = os.path.dirname(os.path.realpath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_subdirectory = os.path.join(log_directory, timestamp)
    os.makedirs(new_subdirectory, exist_ok=True)

    for item in os.listdir(current_directory):
        s = os.path.join(current_directory, item)
        d = os.path.join(new_subdirectory, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)
