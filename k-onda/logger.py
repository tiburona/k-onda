import os
import shutil
from datetime import datetime


def log_directory_contents(log_directory):
    # Get the current file's directory
    current_directory = os.path.dirname(os.path.realpath(__file__))

    # Construct the timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Construct the new subdirectory name
    new_subdirectory = os.path.join(log_directory, timestamp)

    # Create the new subdirectory
    os.makedirs(new_subdirectory, exist_ok=True)

    # Copy the whole contents of the current directory into the log directory
    for item in os.listdir(current_directory):
        s = os.path.join(current_directory, item)
        d = os.path.join(new_subdirectory, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)
