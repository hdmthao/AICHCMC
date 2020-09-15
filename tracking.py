import numpy as np
import pickle
import os
import sys
import json
import time
from tracker.iou_tracker import *

PATH = "/content/HCMCAIC/"
PATH_VIDEO =  os.path.join(PATH, "data/videos")
PATH_BBOX = os.path.join(PATH, "detection_info")

def format_bbox(video_name, file_name):
    ''' prepare formatted bbox for tracking'''

    file_content = open(os.path.join(PATH_BBOX,file_name),'rb')
    content = pickle.load(file_content)
    print("Processing:",video_name)
    data = []

    for fr_id, fr_content in enumerate(content):
        dets = []
        bboxes_type1 = fr_content[1]
        bboxes_type2 = fr_content[2]
        bboxes_type3 = fr_content[3]
        bboxes_type4 = fr_content[4]
        for bb in bboxes_type1:
            score = bb[4]
            dets.append({'bbox': (bb[0], bb[1], bb[2], bb[3]), 'score': score, 'class': 1})
        for bb in bboxes_type2:
            score = bb[4]
            dets.append({'bbox': (bb[0], bb[1], bb[2], bb[3]), 'score': score, 'class': 2})
        for bb in bboxes_type3:
            score = bb[4]
            dets.append({'bbox': (bb[0], bb[1], bb[2], bb[3]), 'score': score, 'class': 3})
        for bb in bboxes_type4:
            score = bb[4]
            dets.append({'bbox': (bb[0], bb[1], bb[2], bb[3]), 'score': score, 'class': 4})
        data.append(dets)
    file_content.close()
    return data

if __name__ == "__main__":
    duration = time.time()
    print("Running tracking on ", PATH_BBOX)
    for file_name in os.listdir(PATH_BBOX):
        if file_name[0] == '.':
          continue
        vid_name = file_name[:-8]
        data = format_bbox(vid_name, file_name)
        content_video_path = os.path.join(PATH_VIDEO, vid_name+".mp4")
        visualize = False
        results = track_iou_edited(vid_name, data, 0.3, 0.7, 0.15, 10, PATH_VIDEO, visualize)

    duration = time.time() - duration 
    print("Total tracking time:", duration)
