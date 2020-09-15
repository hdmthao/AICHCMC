import os
import numpy as np
import sys
from PIL import Image

PATH_MOI = './drive/data/moi'
def load_image_to_np(path):
    return np.array(Image.open(path))

def run():
    cam_name = sys.argv[1]
    for folder_name in os.listdir(PATH_MOI):
        if folder_name != cam_name:
            continue
        for file_name in os.listdir(os.path.join(PATH_MOI, folder_name)):
            if file_name.endswith(".png"):
                movement_id = file_name.split("_")[-1][:-4]
                full_path_name = os.path.join(PATH_MOI, folder_name, file_name)
                frame_np = load_image_to_np(full_path_name)
                np.save(os.path.join(PATH_MOI, folder_name, file_name[:-4]), frame_np)

run()
