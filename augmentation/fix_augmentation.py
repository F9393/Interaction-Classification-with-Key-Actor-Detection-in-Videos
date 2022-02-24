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

org_path = "./hockey_dataset"
aug_path = "./aug_hockey_dataset2"
# body_dir = 'CLEANED_BODY'
SERIES = ["Tripping", "Slashing", "No_penalty"]

def fix_annotation(org_json, aug_json):
    keys = set().union(*(d.keys() for d in org_json))
    keys.remove('frameNum')
    all_players = list(keys)

    new_json = {}

    for org_frame, aug_frame in zip(org_json, aug_json):
        # print(frame)
        for key in all_players:
            org_pose = org_frame[key]
            aug_pose = aug_frame[key]

            for b in range(0,48,3):
                x = org_pose[b]
                y = org_pose[b+1]
                if x == 0.0 and y == 0.0:
                    aug_pose[b] = 0.0
                    aug_pose[b+1] = 0.0

                xa = aug_pose[b]
                ya = aug_pose[b+1]

                if xa < 0:
                    aug_pose[b] = 0.0
                if ya < 0:
                    aug_pose[b+1] = 0.0

        print("done")

    return aug_json


if __name__ == "__main__":

    new_json = {}
    list_num_players = []
    for series_no in SERIES:
        for video in os.listdir(osp.join(org_path, series_no)):
            with open(osp.join(org_path, series_no, video, video + ".json"), 'r') as f1:
                # print(osp.join(annotations_path, series_no, video, video + ".json"))
                org_json = json.load(f1)
            with open(osp.join(aug_path, series_no, video + "200", video + "200" + ".json"), 'r') as f2:
                aug_json = json.load(f2)
                correct_json = fix_annotation(org_json, aug_json)
                print("done")

                with open(
                        join(
                            aug_path, series_no,
                            video + "200",
                            video + "200" + ".json",
                        ),
                        "w",
                ) as fo:
                    json.dump(correct_json, fo, indent=4)
                #all frames padded


