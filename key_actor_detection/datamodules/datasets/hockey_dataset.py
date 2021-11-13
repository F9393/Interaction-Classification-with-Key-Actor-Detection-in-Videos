import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils import data
from tqdm import tqdm
import glob
import pickle
import json
from collections import OrderedDict
import numpy as np

# class HockeyDataset(data.Dataset):
#     def __init__(
#         self,
#         data,
#         resize,
#         stage,
#         num_frames,
#         max_players,
#         num_pose_coordinates,
#         **kwargs,
#     ):
#         """
#         Dataset class containing utility functions to read the image and pose data in SBU Kinect dataset.
#         As the size of SBU Kinect dataset is small, all images and pose values are 
#         read and stored beforeheand to save data loading time. Optionally, this loaded data
#         can also be written to a pickle file to save reading time for future runs.
#         This class does NOT implement __getitem__. New classes must be written that inherit from
#         this class, using the data saved in "self.loaded_videos" inside __getitem__.

#         Parameters
#         ----------
#         set_paths : list
#             paths to the participant sets (e.g '../../s01s02')
#         select_frames : int
#             these number of frames will be selected from the middle of each video
#         stage : string
#             one of "train" or "test"
#         resize: int
#             all frames will be resized to (resize,resize)
#         fold_no: int
#             if using K-fold CV, fold_no is incorporated in name of saved pickle file 
#         data_dir: string
#             dataset path

#         """

#         self.X_dirs = data[0]
#         self.y = data[1]
#         self.resize = resize
#         self.stage = stage
#         self.num_frames = num_frames
#         self.max_players = max_players
#         self.num_pose_coordinates = num_pose_coordinates
#         self.poses = {}

#         # remove every third element from the poses (confidence is always 1)
#         for penalty_dir in self.X_dirs:
#             with open(os.path.join(penalty_dir, 'skeleton.txt'), "r") as f:
#                 self.poses[penalty_dir] = json.loads(f.read(), object_pairs_hook=OrderedDict)
#                 for frame_no in len(self.poses[penalty_dir]):
#                     frame_poses = self.poses[penalty_dir][frame_no]
#                     for player_no in frame_poses.keys():
#                         if player_no.startswith("p"):
#                             del self.poses[penalty_dir][frame_no][player_no][2::3] 

#         if stage == "train":
#             self.transform = transforms.Compose(
#                 [
#                     transforms.Resize([resize, resize]),
#                     transforms.ToTensor(),
#                     transforms.Normalize(
#                         mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
#                     ),
#                 ]
#             )
#         elif stage == "val":
#             self.transform = transforms.Compose(
#                 [
#                     transforms.Resize([resize, resize]),
#                     transforms.ToTensor(),
#                     transforms.Normalize(
#                         mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
#                     ),
#                 ]
#             )            

#     def read_images(self, path):
#         X = []
#         for i in range(self.num_frames):
#             image = Image.open(os.path.join(path, 'frame{:04d}.png'.format(i)))
#             image = self.transform(image)
#             X.append(image)
#         X = torch.stack(X, dim=0)
#         return X

#     def __len__(self):
#         return len(self.X_dirs)

#     def __getitem__(self, index):
#         raise NotImplementedError()

class FrameLoader(data.Dataset):
    def __init__(
        self,
        data,
        resize,
        stage,
        num_frames,
        **kwargs,
    ):

        self.X_dirs = data[0]
        self.y = data[1]
        self.resize = resize
        self.stage = stage
        self.num_frames = num_frames

        if stage == "train":
            self.transform = transforms.Compose(
                [
                    transforms.Resize([resize, resize]),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        elif stage == "val":
            self.transform = transforms.Compose(
                [
                    transforms.Resize([resize, resize]),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )            

    def read_images(self, path):
        X = []
        for i in range(self.num_frames):
            image = Image.open(os.path.join(path, 'frame_{:04d}.png'.format(i)))
            image = self.transform(image)
            X.append(image)
        X = torch.stack(X, dim=0)
        return X



class PoseLoader(data.Dataset):
    def __init__(
        self,
        data,
        stage,
        num_frames,
        max_players,
        num_keypoints,
        coords_per_keypoint,
        **kwargs,
    ):

        self.X_dirs = data[0]
        self.y = data[1]
        self.stage = stage
        self.num_frames = num_frames
        self.max_players = max_players
        self.num_keypoints = num_keypoints
        self.coords_per_keypoint = coords_per_keypoint
        self.poses_and_masks = {}

        # Uses only x,y coordinates. Removes every third element from the poses (confidence is always 1).
        for penalty_dir in self.X_dirs:
            self.poses = np.zeros((self.num_frames, self.max_players, self.num_keypoints*self.coords_per_keypoint), dtype='float32')
            self.mask =  np.ones((self.num_frames, self.max_players, self.num_keypoints*self.coords_per_keypoint), dtype='float32')
            with open(os.path.join(penalty_dir, 'skeleton.json'), "r") as f:
                tmp_poses = json.loads(f.read(), object_pairs_hook=OrderedDict)
                for frame_no in range(self.num_frames):
                    frame_poses = tmp_poses[frame_no]
                    for player_no,player_pose in frame_poses.items():
                        if player_no.startswith("p"):
                            del player_pose[2::3]     
                            self.poses[frame_no, int(player_no[1:]), :] = player_pose
                            self.mask[frame_no, int(player_no[1:]), :] = 0 # if mask=0, means do not mask these values
                self.poses_and_masks[penalty_dir] = (self.poses, self.mask)            

class M1_HockeyDataset(FrameLoader):
    """
    dataloader for model 1.
    """

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        penalty_dir = self.X_dirs[index]

        # Load data
        X = self.read_images(penalty_dir)     # (input) spatial images
        y = torch.LongTensor([self.y[index]])                  # (labels) LongTensor are for int64 instead of FloatTensor
        return X, y


class M2_HockeyDataset(PoseLoader):
    """
    dataloader for model 2
    """

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        penalty_dir = self.X_dirs[index]
        padded_poses = self.poses_and_masks[penalty_dir][0]
        mask = self.poses_and_masks[penalty_dir][1]
        masked_poses = np.ma.array(padded_poses, mask=mask)
        averaged_poses = np.mean(masked_poses, 1)
        return np.ma.getdata(averaged_poses).astype('float32'), self.y[index]

class M3_HockeyDataset(PoseLoader):
    """
    dataloader for model 3
    """

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        penalty_dir = self.X_dirs[index]
        padded_poses = self.poses_and_masks[penalty_dir][0]
        mask = self.poses_and_masks[penalty_dir][1]

        # padded_poses, mask : (#frames, max_players, #keypoints_per_player)
        return padded_poses, mask, self.y[index]


class M4_HockeyDataset(M2_HockeyDataset):
    """
    dataloader for model 4.
    """
    def __init__(
        self,
        data,
        resize,
        stage,
        num_frames,
        max_players,
        num_keypoints,
        **kwargs,
    ):

        FrameLoader.__init__(data, resize, stage, num_frames)
        PoseLoader.__init__(data,stage,num_frames,max_players, num_keypoints)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        # load frames : shape (10,3,224,224)
        frames = self.loaded_videos['frames'][index]

        # load poses
        pose_values = self.loaded_poses[index]
        pose_values = pose_values.view(
            pose_values.shape[0], 2, self.keypoints_per_person
        )

        y = torch.LongTensor([self.labels[index]])

        # frames : (10,3,224,224) , pose_values : (10,2,30) (if only x,y otherwise (10,2,45))
        return frames, pose_values, y

