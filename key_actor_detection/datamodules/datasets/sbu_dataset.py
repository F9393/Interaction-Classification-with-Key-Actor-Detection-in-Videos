import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils import data
from tqdm import tqdm
import glob
import pickle


class BaseSBUDataset(data.Dataset):
    def __init__(
        self,
        set_paths,
        select_frames,
        stage,
        resize,
        fold_no,
        data_dir,
        cache_folds,
        use_cache,
        folds_cache_path,
        **kwargs,
    ):
        """
        Dataset class containing utility functions to read the image and pose data in SBU Kinect dataset.
        As the size of SBU Kinect dataset is small, all images and pose values are 
        read and stored beforeheand to save data loading time. Optionally, this loaded data
        can also be written to a pickle file to save reading time for future runs.
        This class does NOT implement __getitem__. New classes must be written that inherit from
        this class, using the data saved in "self.loaded_videos" inside __getitem__.

        Parameters
        ----------
        set_paths : list
            paths to the participant sets (e.g '../../s01s02')
        select_frames : int
            these number of frames will be selected from the middle of each video
        stage : string
            one of "train" or "test"
        resize: int
            all frames will be resized to (resize,resize)
        fold_no: int
            if using K-fold CV, fold_no is incorporated in name of saved pickle file 
        data_dir: string
            dataset path
        cache_folds: bool
            whether the loaded data be pickled and saved into folds_cache_path
        use_cache: bool
            whether to use previously saved pickle file, skipping reading images and pose from dataset 
        folds_cache_path: string
            if cache_folds is true, the training and val sets for this particular fold will be pickled and 
            saved into folds_cache_path

        """

        self.data_dir = data_dir
        self.select_frames = select_frames
        self.stage = stage
        self.loaded_videos = None
        self.fold_no = fold_no
        self.resize = resize

        if stage == "train" or stage == "val":
            self.transform = transforms.Compose(
                [
                    transforms.Resize([resize, resize]),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

        self.folders, self.labels, self.video_len = list(
            zip(*self._get_video_metdata(set_paths))
        )

        loaded_dataset_path = os.path.join(
            folds_cache_path, f"sbu_{stage}_fold={fold_no}.pkl"
        )
        if os.path.exists(loaded_dataset_path) and use_cache:
            print(f"Using cached {stage} data.")
            with open(loaded_dataset_path, "rb") as f:
                self.loaded_videos = pickle.load(f)
        else:
            if cache_folds and not os.path.exists(folds_cache_path):
                raise Exception(
                    f"Directory {folds_cache_path} not found! Create directory and re-run."
                )

            self.loaded_videos = self._load_videos()

            if os.getenv("SLURM_LOCALID", '0') == '0' and cache_folds:
                with open(loaded_dataset_path, "wb") as f:
                    print(f"Writing {stage} fold {fold_no} to cache.")
                    pickle.dump(self.loaded_videos, f)

    def _get_video_metdata(self, set_paths):
        """
        Gets meta data from the dataset.

        Parameters
        ----------
        set_paths : list
            paths to the participant sets (e.g '../../s01s02')

        Returns
        -------
        output : list [(video_path, class label, number of frames in the video)]
            video_path : e.g '../../s01s02/01/001'
            class_label : integer from [0,7]
            number of frames : number of frames in this video

            video_paths are found by walking through set_paths

        """

        all_data = []

        for part_path in set_paths:
            cats = sorted([s.decode("utf-8") if type(s) == bytes else s for s in os.listdir(part_path)])[1:]
            for cat in cats:
                label = int(cat) - 1
                cat_path = os.path.join(part_path, cat)  # path to category(1-8)
                runs = sorted(os.listdir(cat_path))
                if runs[0] == ".DS_Store":
                    runs = runs[1::]
                for run in runs:
                    run_path = os.path.join(cat_path, run)
                    num_frames = len(glob.glob(f"{run_path}/rgb*"))
                    all_data.append((run_path, label, num_frames))
        return all_data

    def _load_videos(self):
        """
        Reads RGB images and pose data. Folders to read from are found from meta-data.

        Parameters
        ----------
        None

        Returns
        -------
        output : list [(loaded_frames, video_pose_values)]
            loaded_frames : 4D Tensor of shape (#frames, #channels, resize, resize)
            video_pose_values : 2D Tensor of shape (#frames, 90) containing normalized x,y,z pose coordinates for each of 
            15 keypoints for both participants in each frame. First 45 values correspond to participant 1 and next 45 to 
            participant 2.

        """

        # load image data
        loaded_videos = []
        with tqdm(self.folders) as pbar:
            pbar.set_description(f"Reading {self.stage} fold {self.fold_no} data!")
            for index, video_pth in enumerate(pbar):
                video_len = self.video_len[index]
                start_idx = (video_len - self.select_frames) // 2
                end_idx = start_idx + self.select_frames

                frame_pths = sorted(glob.glob(f"{video_pth}/rgb*"))
                reqd_frame_pths = frame_pths[start_idx:end_idx]
                loaded_frames = []
                loaded_frames = torch.zeros(
                    self.select_frames, 3, self.resize, self.resize, dtype=torch.float32
                )
                for idx, frame_pth in enumerate(reqd_frame_pths):
                    frame = Image.open(frame_pth).convert("RGB")
                    frame = (
                        self.transform(frame) if self.transform is not None else frame
                    )  # impose transformation if exists
                    loaded_frames[idx] = frame

                # load pose data
                with open(os.path.join(video_pth, "skeleton_pos.txt"), "r") as f:
                    pose_data = f.readlines()

                assert len(frame_pths) == len(pose_data), "pose data loaded incorrectly"

                reqd_pose_data = pose_data[start_idx:end_idx]

                video_pose_values = []
                for row in reqd_pose_data:
                    posture_data = [x.strip() for x in row.split(",")]
                    frame_pose_values = [float(x) for x in posture_data[1:]]
                    assert (
                        len(frame_pose_values) == 90
                    ), "incorrect number of pose values"
                    video_pose_values.append(frame_pose_values)
                video_pose_values = torch.tensor(video_pose_values, dtype=torch.float32)

                # loaded_frames : (10,3,224,224), video_pose_values : (10,90)
                loaded_videos.append((loaded_frames, video_pose_values))

        assert len(loaded_videos) == len(
            self.folders
        ), "error in reading images of videos"

        return loaded_videos

    def __len__(self):
        return len(self.loaded_videos)

    def __getitem__(self, index):
        raise NotImplementedError()


class M1_SBU_Dataset(BaseSBUDataset):
    """
    dataloader for model 1.
    """

    def __len__(self):
        return len(self.loaded_videos)

    def __getitem__(self, index):
        X = self.loaded_videos[index][0]
        y = torch.LongTensor([self.labels[index]])
        return X, y


class M2_SBU_Dataset(BaseSBUDataset):
    """
    dataloader for model 2
    """

    def __init__(self, dim, *args, **kwargs):
        super(M2_SBU_Dataset, self).__init__(*args, **kwargs)
        self.keypoints_per_person = None
        self.loaded_poses = None

        if dim == 2:
            idxs = torch.tensor([i for i in range(90) if (i + 1) % 3 != 0])
            self.loaded_poses = [x[1][:, idxs] for x in self.loaded_videos]
            self.keypoints_per_person = 30
        elif dim == 3:
            self.loaded_poses = [x[1] for x in self.loaded_videos]
            self.keypoints_per_person = 45
        else:
            raise Exception("invalid dim in M2_SBU_Dataset! Must be either 2 or 3.")

    def __getitem__(self, index):
        pose_values = self.loaded_poses[index]
        pose_values = pose_values.view(-1, 2, self.keypoints_per_person)
        avg_pose = torch.mean(
            pose_values, 1
        )  # if dim=2, shape = (T,30) else shape = (T,45)
        y = torch.LongTensor([self.labels[index]])

        return avg_pose, y


class M3_SBU_Dataset(M2_SBU_Dataset):
    """
    dataloader for model 3
    """

    def __getitem__(self, index):
        pose_values = self.loaded_poses[index]
        pose_values = pose_values.view(
            pose_values.shape[0], 2, self.keypoints_per_person
        )
        y = torch.LongTensor([self.labels[index]])

        # pose_values : (#frames, #players, #keypoints_per_player)
        return pose_values, y


class M4_SBU_Dataset(M2_SBU_Dataset):
    """
    dataloader for model 4.
    """

    def __getitem__(self, index):
        # load frames : shape (10,3,224,224)
        frames = self.loaded_videos[index][0]

        # load poses
        pose_values = self.loaded_poses[index]
        pose_values = pose_values.view(
            pose_values.shape[0], 2, self.keypoints_per_person
        )

        y = torch.LongTensor([self.labels[index]])

        # frames : (10,3,224,224) , pose_values : (10,2,30) (if only x,y otherwise (10,2,45))
        return frames, pose_values, y

