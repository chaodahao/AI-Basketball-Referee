import os

BALL_MODEL_PATH = "./lib/yolov5/checkpoint/yolov5s_basketball.pt"
# BALL_MODEL_PATH = "./checkpoints/basketballModel.pt"
POSE_MODEL_PATH = "./checkpoints/yolov8s-pose.pt"

VIDEO = "./videos/c.mp4"
SAVE_BASE_PATH = "./travel_footage"

SAVING = True

def check_files(file_list):
    """
    Check the existence of files in the given list.
    If any file in the list does not exist, an AssertionError is raised with a corresponding error message.

    Parameters:
    file_list (list): A list of file paths to be checked.
    """
    
    for f in file_list:
        assert os.path.exists(f), f"{f} do not exists, please check whether the file name or file path given is correct"
        print(f"{f} exists")
    
def mk_dir(file_list):
    """
    Create directories in the given list if they do not already exist.

    Parameters:
    file_list (list): A list of directory paths to be created.
    """
    for f in file_list:
        if not os.path.exists(f):
            print(f"Directory:{f} do not exists, already created!")
            os.mkdir(f)

check_files([BALL_MODEL_PATH, POSE_MODEL_PATH, VIDEO])
mk_dir([SAVE_BASE_PATH])