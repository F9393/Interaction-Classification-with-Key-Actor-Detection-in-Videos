import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils import data
import json
from collections import OrderedDict
import numpy as np

class FrameReader():
    def __init__(
        self,
        resize,
        stage,
        num_frames,
        **kwargs,
    ):

        self.resize = resize
        self.stage = stage
        self.num_frames = num_frames

        if stage == "train" or stage == "val" or stage == "test":
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

class PoseReader():
    def __init__(
        self,
        X_dirs,
        stage,
        num_frames,
        max_players,
        num_keypoints,
        coords_per_keypoint,
        **kwargs,
    ):

        self.X_dirs = X_dirs
        self.stage = stage
        self.num_frames = num_frames
        self.max_players = max_players
        self.num_keypoints = num_keypoints
        self.coords_per_keypoint = coords_per_keypoint
        self.poses_and_masks = self.generate_pose_and_masks()

    def generate_pose_and_masks(self):
        """
        Uses only x,y coordinates. Removes every third element from the poses (confidence is always 1).
        Returns tuple(pose:[64,15,40], mask:[64,15,40]) i.e #frames,#max_players,#keypoint values
        """
        
        poses_and_masks = {}
        for penalty_dir in self.X_dirs:
            self.poses = np.zeros((self.num_frames, self.max_players, self.num_keypoints*self.coords_per_keypoint), dtype='float32')
            self.mask =  np.ones((self.num_frames, self.max_players, self.num_keypoints*self.coords_per_keypoint), dtype='float32')
            with open(os.path.join(penalty_dir, f'{os.path.basename(penalty_dir)}.json'), "r") as f:
                tmp_poses = json.loads(f.read(), object_pairs_hook=OrderedDict)
                # print(penalty_dir)
                for frame_no in range(self.num_frames):
                    # print(frame_no)
                    frame_poses = tmp_poses[frame_no]
                    for player_no,player_pose in frame_poses.items():
                        # print(player_no)
                        if player_no.startswith("p"):
                            del player_pose[2::3]     
                            self.poses[frame_no, int(player_no[1:]), :] = player_pose
                            self.mask[frame_no, int(player_no[1:]), :] = 0 # if mask=0, means do not mask these values
                self.poses = np.delete(self.poses, obj=0, axis=1)
                self.mask = np.delete(self.mask, obj=0, axis=1)
                poses_and_masks[penalty_dir] = (self.poses, self.mask)

        if self.stage == 'test':

            new_dict_test = {}
            for key, val in poses_and_masks.items():
                pose_list = val[0].tolist()
                new_dict_test[key] = pose_list
            with open("/home/fay/Desktop/Key-Actor-Detection/work_dir/test_stage.json", "w") as f:
                json.dump(new_dict_test, f, indent=4)

        return poses_and_masks        

class M1_HockeyDataset(data.Dataset):
    """
    dataloader for model 1.
    """

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
        self.frame_reader = FrameReader(resize, stage, num_frames, **kwargs)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        penalty_dir = self.X_dirs[index]

        # Load data
        X = self.frame_reader.read_images(penalty_dir)    
        y = torch.LongTensor([self.y[index]])                  
        return X, y


class M2_HockeyDataset(data.Dataset):
    """
    dataloader for model 2
    """

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
        self.pose_reader = PoseReader(self.X_dirs, stage, num_frames, max_players, num_keypoints, coords_per_keypoint, **kwargs)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        penalty_dir = self.X_dirs[index]
        padded_poses = self.pose_reader.poses_and_masks[penalty_dir][0]
        mask = self.pose_reader.poses_and_masks[penalty_dir][1]
        masked_poses = np.ma.array(padded_poses, mask=mask)
        averaged_poses = np.mean(masked_poses, 1)
        return np.ma.getdata(averaged_poses).astype('float32'), self.y[index]

class M3_HockeyDataset(data.Dataset):
    """
    dataloader for model 3
    """
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
        self.pose_reader = PoseReader(self.X_dirs, stage, num_frames, max_players, num_keypoints, coords_per_keypoint, **kwargs)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        penalty_dir = self.X_dirs[index]
        padded_poses = self.pose_reader.poses_and_masks[penalty_dir][0]
        mask = self.pose_reader.poses_and_masks[penalty_dir][1]

        # padded_poses, mask : (#frames, max_players, #keypoints_per_player)
        return padded_poses, mask, self.y[index]


class M4_HockeyDataset(data.Dataset):
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
        coords_per_keypoint,
        **kwargs,
    ):

        self.X_dirs = data[0]
        self.y = data[1]
        self.frame_reader = FrameReader(resize, stage, num_frames, **kwargs)
        self.pose_reader = PoseReader(self.X_dirs, stage, num_frames, max_players, num_keypoints, coords_per_keypoint, **kwargs)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        # load frames : shape (64,3,224,224)
        penalty_dir = self.X_dirs[index]
        frames = self.frame_reader.read_images(penalty_dir)     # (input) spatial images

        # load poses
        padded_poses = self.pose_reader.poses_and_masks[penalty_dir][0]
        mask = self.pose_reader.poses_and_masks[penalty_dir][1]

        y = torch.LongTensor([self.y[index]])

        # frames : (10,3,224,224) , pose_values : (10,2,30) (if only x,y otherwise (10,2,45))
        return frames, padded_poses, mask, y

