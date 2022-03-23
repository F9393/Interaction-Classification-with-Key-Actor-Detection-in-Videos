import json
import os
import os.path as osp
from os.path import join

org_path = "./hockey_dataset"
out_path = "./No_stick_hockey_dataset"
SERIES = ["Tripping", "Slashing", "No_penalty"]


def remove_stick_func(org_json):

    """
    Script to remove the hockey stick annotations for ablation studies
    """

    keys = set().union(*(d.keys() for d in org_json))
    keys.remove("frameNum")
    all_players = list(keys)

    for org_frame in org_json:
        for key in all_players:
            org_pose = org_frame[key]
            # number of keupoints are known and given
            org_pose = org_pose[0:42]
            org_frame[key] = org_pose

    return org_json


if __name__ == "__main__":

    new_json = {}
    list_num_players = []

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for series_no in SERIES:
        dest_sub_folder = os.path.join(out_path, series_no)
        if not os.path.exists(dest_sub_folder):
            os.makedirs(dest_sub_folder)
        for video in os.listdir(osp.join(org_path, series_no)):
            with open(osp.join(org_path, series_no, video, video + ".json"), "r") as f1:
                org_json = json.load(f1)
                no_stick_json = remove_stick_func(org_json)

                dest_frame_pose_path = os.path.join(dest_sub_folder, video)
                if not os.path.exists(dest_frame_pose_path):
                    os.makedirs(dest_frame_pose_path)

                with open(
                    join(
                        out_path,
                        series_no,
                        video,
                        video + ".json",
                    ),
                    "w",
                ) as fo:
                    json.dump(no_stick_json, fo, indent=4)
