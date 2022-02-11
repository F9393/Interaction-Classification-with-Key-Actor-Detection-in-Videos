import json
from datetime import datetime
import os
import os.path as osp
from os import listdir, makedirs
from os.path import isdir, join
import warnings
import shutil
import sys
import argparse
import re

#set all confidences to one

annotations_path = "./"
body_dir = 'CLEANED_BODY'
SERIES = [5]
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


if __name__ == "__main__":

    for series_no in SERIES:
        for cat in os.listdir(osp.join(annotations_path, f'series_{series_no}')):
            cleaned_body_dir = osp.join(annotations_path, f'series_{series_no}', cat, 'CLEANED_BODY')
            output_dir = osp.join(annotations_path, f'series_{series_no}', cat, 'CLEANED_BODY')

            if isdir(cleaned_body_dir):
                cleaned_body_clip_names = os.listdir(cleaned_body_dir)
            else:
                raise Exception("Filtered Poses must contain some clips!")

            cleaned_body_clip_dir_paths = [osp.join(cleaned_body_dir, x) for x in
                                           cleaned_body_clip_names]  # e.g 'hockey_dataset/series_1/No_penalty/CLEANED_BODY/_2017-11-16-wsh-col-home570'

            AddZero(cleaned_body_clip_dir_paths)

    # out_file.close()
