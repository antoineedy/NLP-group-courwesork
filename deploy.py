# import os

# print("-- Step 1: launching the web app locally --")
# os.system("python app.py")
# print("-- Step 1 DONE --")

# print("-- Step 2: running simple tests --")
# os.system("python utils/simple_test.py")
# print("-- Step 2 DONE --")

# print("-- Step 3: running stress tests --")
# os.system("python utils/stress_test.py")
# print("-- Step 3 DONE --")

import os
import subprocess
import platform
import time


def open_in_new_terminal(command):
    system = platform.system()
    if system == "Linux":
        # For Linux, using gnome-terminal (you might need to adjust for other terminal emulators)
        subprocess.Popen(["gnome-terminal", "--", "bash", "-c", command])
    elif system == "Darwin":
        # For macOS
        script = f"""
        tell application "Terminal"
            do script "{command}"
            activate
            set currentTab to do script "{command}"
        end tell
        """
        subprocess.Popen(["osascript", "-e", script])
    elif system == "Windows":
        # For Windows
        subprocess.Popen(["start", "cmd", "/k", command], shell=True)
    else:
        raise OSError(f"Unsupported operating system: {system}")


print("################# Step 1: launching the web app locally #################")
# find the location of the current directory
current_dir = os.path.dirname(os.path.realpath(__file__))
open_in_new_terminal(f"cd {current_dir} && source bin/activate && python app.py")
print("################# Step 1 DONE #################")

print("################# Step 2: running simple tests #################")
os.system("python utils/simple_test.py")
print("################# Step 2 DONE #################")

print("################# Step 3: running stress tests #################")
os.system("python utils/stress_test.py")
print("################# Step 3 DONE #################")
