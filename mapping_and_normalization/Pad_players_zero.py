import json
from datetime import datetime
import os
import os.path as osp
from os import listdir, makedirs
from os.path import isdir, join
import statistics
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

import warnings
import shutil
import sys
import argparse
import re
import numpy as np

#set all confidences to one

annotations_path = "/home/fay/Desktop/Key-Actor-Detection/hockey_dataset"
# body_dir = 'CLEANED_BODY'
SERIES = ["Tripping", "Slashing", "No_penalty"]
#
# def atoi(text):
#     return int(text) if text.isdigit() else text
#
# def natural_keys(text):
#     return [ atoi(c) for c in re.split(r'(\d+)', text) ]


def AddZero(cleaned_body_clip_dir_paths):

    pose_padding = [0.0,0.0,1.0] * 18
    confidence_pad = [1.0] * 18
    for clip_dir_path in cleaned_body_clip_dir_paths:
        clip_dir = osp.basename(clip_dir_path) # e.g '_2017-11-16-wsh-col-home570'

        # load the body annotations
        with open(
            join(
                clip_dir_path,
                clip_dir + ".json",
            ),
            "r",
        ) as f:
            body = json.load(f)

        keys = set().union(*(d.keys() for d in body))
        keys.remove('frameNum')
        all_players = list(keys)
        # all_players.sort(key=natural_keys)
        print(clip_dir)
        print(all_players)

        for frame in body:
            for key in all_players:
                if key in frame.keys():
                    # print(frame)
                    #set confidence to float zero
                    frame[key][2::3] = confidence_pad
                else:
                    frame[key] = pose_padding

        with open(
                join(
                    clip_dir_path,
                    "padded" + clip_dir + ".json",
                ),
                "w",
        ) as fo:
            json.dump(body, fo, indent=4)

def head_mapping(org_json):
    keys = set().union(*(d.keys() for d in org_json))
    keys.remove('frameNum')
    all_players = list(keys)

    new_json = {}

    for frame in org_json:
        # print(frame)
        for key in all_players:
            org_pose = frame[key]
            head_points = org_pose[0:3]
            head_points = head_points + org_pose[42:54]
            body = org_pose[3:42] + org_pose[54:]
            nose = head_points[0:2]
            reye = head_points[3:5]
            leye = head_points[6:8]
            rear = head_points[9:11]
            lear = head_points[12:14]
            body_list = [nose, reye, leye, rear, lear]

            count = 0
            zero_count = 0
            x = 0
            y = 0

            for parts in body_list:
                if any(parts):
                    count = count + 1
                    x = x + parts[0]
                    y = y + parts[1]
                else:
                    zero_count = zero_count + 1

            #if zero_count == 5 or count_o + zero_count == 5:
            if zero_count == 5:
                new_key_point = [0.0, 0.0, 1.0]
            else:
                if count == 0:
                    print("error")
                else:
                    new_key_point_x = x / count
                    new_key_point_y = y / count
                    new_key_point = [new_key_point_x, new_key_point_y, 1.0]
            # print(new_key_point)
            # print("done")
            org_pose = new_key_point + body
            frame[key] = org_pose
        #print("done")

    return org_json

def get_stats(video, org_json, list_num_players):

    num_players = len(org_json[0]) - 1
    if num_players == 8:
        print(video)
    list_num_players.append((num_players))

    return list_num_players

def remove_zeros(org_json):

    #remove the near zero keypoints

    keys = set().union(*(d.keys() for d in org_json))
    keys.remove('frameNum')
    all_players = list(keys)
    pad = [0.0,0.0,1.0] * 20

    for frame in org_json:
        # print(frame)
        for key in all_players:
            if key not in frame.keys():
                frame[key] = pad
            org_pose = frame[key]
            # print(org_pose)
            for b in range(0,54,3):
                x = org_pose[b]
                y = org_pose[b+1]
                if 0.0 <= x < 50.0 and 0.0 <= y < 50.0:
                    # print("here")
                    org_pose[b] = 0
                    org_pose[b+1] = 0
            # print(org_pose)
            #print("player done")

    return org_json

if __name__ == "__main__":

    new_json = {}
    list_num_players = []
    for series_no in SERIES:
        for video in os.listdir(osp.join(annotations_path, series_no)):
            with open(osp.join(annotations_path, series_no, video, video + ".json"), 'r') as f:
                # print(osp.join(annotations_path, series_no, video, video + ".json"))
                org_json = json.load(f)
                zero_remove_json = remove_zeros(org_json)
                new_body_json = head_mapping(zero_remove_json)
                out = get_stats(video, new_body_json, list_num_players)

                with open(
                        join(
                            annotations_path, series_no,
                            video,
                            video + ".json",
                        ),
                        "w",
                ) as fo:
                    json.dump(new_body_json, fo, indent=4)
                #all frames padded

    C = Counter(list_num_players)
    print(C)
    weights = np.ones_like(list_num_players) / len(list_num_players)
    plt.hist(list_num_players, weights=weights, bins=15)  # density=False would make counts
    plt.ylabel('count')
    plt.xlabel('Number of Players')
    plt.show()






            # output_dir = osp.join(annotations_path, f'series_{series_no}', cat, 'CLEANED_BODY')

            # AddZero(cleaned_body_clip_dir_paths)

    # out_file.close()
