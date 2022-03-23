import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils import data
from tqdm import tqdm
import glob
import pickle


class Cache:
    def __init__(self, do_cache: bool, use_cache: bool):
        # if both do_cache and use_cache are True, cache will be read and same cache will be written back again
        self.do_cache = do_cache
        self.use_cache = use_cache
        self.read_from_cache = False

    def read(self, cache_path, message=None):
        if self.do_cache and not os.path.exists(os.path.dirname(cache_path)):
            raise Exception(
                f"Directory {os.path.dirname(cache_path)} not found! Create directory and re-run."
            )
        if not self.use_cache or not os.path.exists(cache_path):
            return None
        with open(cache_path, "rb") as f:
            if message is not None:
                print(message)
            cache_item = pickle.load(f)
            self.read_from_cache = True

        return cache_item

    def write(self, cache_item, cache_path, message=None):
        if (
            not self.do_cache
            or os.getenv("SLURM_LOCALID", "0") != "0"
            or self.read_from_cache
        ):  # write cache only on first gpu process
            return None
        else:
            with open(cache_path, "wb") as f:
                if message is not None:
                    print(message)
                pickle.dump(cache_item, f)


def _get_video_metadata(set_paths):
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
        cats = sorted(
            [
                s.decode("utf-8") if type(s) == bytes else s
                for s in os.listdir(part_path)
            ]
        )[1:]
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


def read_images(
    folders,
    videos_len,
    num_frames,
    stage,
    resize,
    fold_no,
    cache_folds,
    use_cache,
    folds_cache_path,
    **kwargs,
):
    def _read_images():
        all_video_frames = torch.zeros(len(folders), num_frames, 3, resize, resize)

        if stage == "train" or stage == "val":
            transform = transforms.Compose(
                [
                    transforms.Resize([resize, resize]),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

        with tqdm(folders) as pbar:
            pbar.set_description(f"Reading {stage} fold {fold_no} images!")

            for index, video_pth in enumerate(pbar):
                video_len = videos_len[index]
                start_idx = (video_len - num_frames) // 2
                end_idx = start_idx + num_frames

                frame_pths = sorted(glob.glob(f"{video_pth}/rgb*"))
                reqd_frame_pths = frame_pths[start_idx:end_idx]
                loaded_frames = torch.zeros(
                    num_frames, 3, resize, resize, dtype=torch.float32
                )
                for idx, frame_pth in enumerate(reqd_frame_pths):
                    frame = Image.open(frame_pth).convert("RGB")
                    frame = transform(frame) if transform is not None else frame
                    loaded_frames[idx] = frame

                all_video_frames[index] = loaded_frames

        return all_video_frames

    cacher = Cache(cache_folds, use_cache)
    cache_path = os.path.join(
        folds_cache_path, f"sbu_{stage}_fold={fold_no}_images.pkl"
    )

    loaded_videos = cacher.read(
        cache_path, f"Reading {stage} fold={fold_no} images from cache"
    )
    if loaded_videos is None:
        loaded_videos = _read_images()
    cacher.write(
        loaded_videos, cache_path, f"Writing {stage} fold={fold_no} images as cache"
    )

    return loaded_videos


def read_poses(
    folders,
    videos_len,
    num_frames,
    stage,
    fold_no,
    cache_folds,
    use_cache,
    folds_cache_path,
    **kwargs,
):
    def _read_poses():
        all_video_poses = torch.zeros(len(folders), num_frames, 90)
        with tqdm(folders) as pbar:
            pbar.set_description(f"Reading {stage} fold {fold_no} poses!")

            for index, video_pth in enumerate(pbar):
                video_len = videos_len[index]
                start_idx = (video_len - num_frames) // 2
                end_idx = start_idx + num_frames

                # load pose data
                with open(os.path.join(video_pth, "skeleton_pos.txt"), "r") as f:
                    pose_data = f.readlines()

                reqd_pose_data = pose_data[start_idx:end_idx]

                video_pose_values = torch.zeros(num_frames, 90, dtype=torch.float32)
                for frame_idx, row in enumerate(reqd_pose_data):
                    posture_data = [x.strip() for x in row.split(",")]
                    frame_pose_values = torch.tensor(
                        [float(x) for x in posture_data[1:]]
                    )
                    assert (
                        len(frame_pose_values) == 90
                    ), "incorrect number of pose values"

                    video_pose_values[frame_idx] = frame_pose_values

                all_video_poses[index] = video_pose_values

        return all_video_poses

    cacher = Cache(cache_folds, use_cache)
    cache_path = os.path.join(folds_cache_path, f"sbu_{stage}_fold={fold_no}_poses.pkl")

    loaded_poses = cacher.read(
        cache_path, f"Reading {stage} fold={fold_no} poses from cache"
    )
    if loaded_poses is None:
        loaded_poses = _read_poses()
    cacher.write(
        loaded_poses, cache_path, f"Writing {stage} fold={fold_no} poses as cache"
    )

    return loaded_poses


class M1_SBU_Dataset(data.Dataset):
    """
    dataloader for model 1.
    """

    def __init__(
        self,
        set_paths,
        num_frames,
        stage,
        resize,
        fold_no,
        cache_folds,
        use_cache,
        folds_cache_path,
        **kwargs,
    ):

        # set_paths = [set_paths[0]]

        self.folders, self.labels, self.videos_len = list(
            zip(*_get_video_metadata(set_paths))
        )

        self.loaded_videos = read_images(
            self.folders,
            self.videos_len,
            num_frames,
            stage,
            resize,
            fold_no,
            cache_folds,
            use_cache,
            folds_cache_path,
        )

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, index):
        X = self.loaded_videos[index]
        y = torch.LongTensor([self.labels[index]])
        return X, y


class M2_SBU_Dataset(data.Dataset):
    """
    dataloader for model 2
    """

    def __init__(
        self,
        set_paths,
        num_frames,
        stage,
        coords_per_keypoint,
        fold_no,
        cache_folds,
        use_cache,
        folds_cache_path,
        *args,
        **kwargs,
    ):

        self.folders, self.labels, self.videos_len = list(
            zip(*_get_video_metadata(set_paths))
        )

        self.keypoints_per_person = None
        self.loaded_poses = read_poses(
            self.folders,
            self.videos_len,
            num_frames,
            stage,
            fold_no,
            cache_folds,
            use_cache,
            folds_cache_path,
        )

        if coords_per_keypoint == 2:
            idxs = torch.tensor([i for i in range(90) if (i + 1) % 3 != 0])
            self.loaded_poses = [x[:, idxs] for x in self.loaded_poses]
            self.keypoints_per_person = 30
        elif coords_per_keypoint == 3:
            self.keypoints_per_person = 45
        else:
            raise Exception(
                "invalid coords_per_keypoint in M2_SBU_Dataset! Must be either 2 or 3."
            )

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, index):
        pose_values = self.loaded_poses[index]
        pose_values = pose_values.view(-1, 2, self.keypoints_per_person)
        avg_pose = torch.mean(
            pose_values, 1
        )  # if coords_per_keypoint=2, shape = (T,30) else shape = (T,45)
        y = torch.LongTensor([self.labels[index]])

        return avg_pose, y


class M3_SBU_Dataset(data.Dataset):
    """
    dataloader for model 3
    """

    def __init__(
        self,
        set_paths,
        num_frames,
        stage,
        coords_per_keypoint,
        fold_no,
        cache_folds,
        use_cache,
        folds_cache_path,
        *args,
        **kwargs,
    ):

        self.folders, self.labels, self.videos_len = list(
            zip(*_get_video_metadata(set_paths))
        )

        self.keypoints_per_person = None
        self.loaded_poses = read_poses(
            self.folders,
            self.videos_len,
            num_frames,
            stage,
            fold_no,
            cache_folds,
            use_cache,
            folds_cache_path,
        )

        if coords_per_keypoint == 2:
            idxs = torch.tensor([i for i in range(90) if (i + 1) % 3 != 0])
            self.loaded_poses = [x[:, idxs] for x in self.loaded_poses]
            self.keypoints_per_person = 30
        elif coords_per_keypoint == 3:
            self.keypoints_per_person = 45
        else:
            raise Exception(
                "invalid coords_per_keypoint in M2_SBU_Dataset! Must be either 2 or 3."
            )

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, index):
        pose_values = self.loaded_poses[index]
        pose_values = pose_values.view(
            pose_values.shape[0], 2, self.keypoints_per_person
        )
        y = torch.LongTensor([self.labels[index]])

        # pose_values : (#frames, #players, #keypoints_per_player)
        return pose_values, y


class M4_SBU_Dataset(data.Dataset):
    """
    dataloader for model 4.
    """

    def __init__(
        self,
        set_paths,
        num_frames,
        stage,
        resize,
        coords_per_keypoint,
        fold_no,
        cache_folds,
        use_cache,
        folds_cache_path,
        *args,
        **kwargs,
    ):

        self.folders, self.labels, self.videos_len = list(
            zip(*_get_video_metadata(set_paths))
        )

        self.keypoints_per_person = None

        self.loaded_videos = read_images(
            self.folders,
            self.videos_len,
            num_frames,
            stage,
            resize,
            fold_no,
            cache_folds,
            use_cache,
            folds_cache_path,
        )
        self.loaded_poses = read_poses(
            self.folders,
            self.videos_len,
            num_frames,
            stage,
            fold_no,
            cache_folds,
            use_cache,
            folds_cache_path,
        )

        if coords_per_keypoint == 2:
            idxs = torch.tensor([i for i in range(90) if (i + 1) % 3 != 0])
            self.loaded_poses = [x[:, idxs] for x in self.loaded_poses]
            self.keypoints_per_person = 30
        elif coords_per_keypoint == 3:
            self.keypoints_per_person = 45
        else:
            raise Exception(
                "invalid coords_per_keypoint in M2_SBU_Dataset! Must be either 2 or 3."
            )

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, index):
        # load frames : shape (10,3,224,224)
        frames = self.loaded_videos[index]

        # load poses
        pose_values = self.loaded_poses[index]
        pose_values = pose_values.view(
            pose_values.shape[0], 2, self.keypoints_per_person
        )

        y = torch.LongTensor([self.labels[index]])

        # frames : (10,3,224,224) , pose_values : (10,2,30) (if only x,y otherwise (10,2,45))
        return frames, pose_values, y
