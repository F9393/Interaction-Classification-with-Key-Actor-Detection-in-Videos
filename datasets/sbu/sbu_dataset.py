import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils import data
from tqdm import tqdm
import glob
import pickle


folds_cache_path = "/usr/local/data01/rohitram/sbu-snapshots/folds-path"
if not os.path.exists(folds_cache_path):
    os.makedirs(folds_cache_path)

class SBU_Dataset(data.Dataset):

    def __init__(self, set_paths, select_frames, mode, resize = 224, fold_no = None):
        """
        Args
            set_paths : paths to the participant sets (e.g '../../s0102')
            select_frames : no. of center frames to use 
            mode : train/test
            transform : set of transformations to perform after reading image

        As the size of sbu dataset is small, all images are read and stored initially to save data loading time 
        """

        self.folders, self.labels, self.video_len = list(zip(*self.get_video_data(set_paths)))
        self.select_frames = select_frames   
        self.mode = mode
        self.loaded_videos = None
        self.fold_no = fold_no
        self.transform = transforms.Compose([transforms.Resize([resize, resize]),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        if mode == 'train':
            if fold_no is None:
                loaded_dataset_path = os.path.join(folds_cache_path, f'sbu_train.pkl')
            else:
                loaded_dataset_path = os.path.join(folds_cache_path, f'sbu_train_fold={fold_no}.pkl')
            if os.path.exists(loaded_dataset_path):
                with open(loaded_dataset_path, 'rb') as f:
                    self.loaded_videos = pickle.load(f)
        if mode == 'valid':
            if fold_no is None:
                loaded_dataset_path = os.path.join(folds_cache_path, f'sbu_valid.pkl')
            else:
                loaded_dataset_path = os.path.join(folds_cache_path, f'sbu_valid_fold={fold_no}.pkl')
            if os.path.exists(loaded_dataset_path):
                with open(loaded_dataset_path, 'rb') as f:
                    self.loaded_videos = pickle.load(f)
        if self.loaded_videos is None:
            self.loaded_videos = self.load_videos()
            with open(loaded_dataset_path, "wb") as f:
                pickle.dump(self.loaded_videos, f)

    def get_video_data(self, set_paths):
        """
        Args
            set_paths : list of paths to participant categories (e.g '../../s01s02') for corresponding set(train/test)

        Returns
            all_data : List[ tuple(video_path, label, num_frames) ].

            Each path in 'set_paths' argument is expanded to generate a list of all paths to videos contained
            in them and the number of frames in each video. Label is category of video (0-7).

        """
    
        all_data = []

        for part_path in set_paths: 
            cats = sorted([s.decode("utf-8") for s in os.listdir(part_path)])[1:]      
            for cat in cats:
                label = int(cat) - 1
                cat_path = os.path.join(part_path,cat) # path to category(1-8)
                runs = sorted(os.listdir(cat_path)) 
                if runs[0] == '.DS_Store':
                    runs = runs[1::]        
                for run in runs:
                    run_path = os.path.join(cat_path, run)
                    num_frames = len(glob.glob(f'{run_path}/rgb*'))
                    all_data.append((run_path,label,num_frames))
        return all_data

    def load_videos(self):
        """
        Args
            -
        
        Returns
            loaded_videos : a list of 4D tensors (select_frames,#channels,img_height,img_width) obtained by reading 'select_frames' frames 
            of all videos in self.folders
        """

        ## load image data
        loaded_videos = []
        print(f'Loading {self.mode} Data!')
        for index, video_pth in enumerate(tqdm(self.folders)):
            video_len = self.video_len[index]
            start_idx = (video_len - self.select_frames) // 2
            end_idx = start_idx + self.select_frames

            frame_pths = sorted(glob.glob(f'{video_pth}/rgb*'))
            reqd_frame_pths = frame_pths[start_idx : end_idx]
            loaded_frames = []
            for frame_pth in reqd_frame_pths:
                frame = Image.open(frame_pth).convert('RGB')
                frame = self.transform(frame) if self.transform is not None else frame  # impose transformation if exists
                loaded_frames.append(frame.squeeze_(0))
            loaded_frames = torch.stack(loaded_frames, dim=0)


            ## load pose data
            with open(os.path.join(video_pth, 'skeleton_pos.txt'), "r") as f:
                pose_data = sorted(f.readlines())

            assert len(frame_pths) == len(pose_data), "pose data loaded incorrectly"

            reqd_pose_data = pose_data[start_idx : end_idx]

            video_pose_values = []
            for row in reqd_pose_data:
                posture_data = [x.strip() for x in row.split(',')]
                frame_pose_values = []
                for i in range(1, len(posture_data), 3):
                    frame_pose_values.extend([float(posture_data[i]), float(posture_data[i+1])]) # 60 dim vector of alternating (x.y) values
                video_pose_values.append(frame_pose_values) # list of 60 dim lists
            video_pose_values = torch.tensor(video_pose_values, dtype=torch.float32)

        
            loaded_videos.append((loaded_frames, video_pose_values))

        assert len(loaded_videos) == len(self.folders), "error in reading images of videos"

        return loaded_videos
    

    def __len__(self):
        return len(self.loaded_videos)

    def __getitem__(self, index):
        raise NotImplementedError()


class M1_SBU_Dataset(SBU_Dataset):
    def __getitem__(self, index):
        X = self.loaded_videos[index][0]   
        y = torch.LongTensor([self.labels[index]])            
        return X, y

class M2_SBU_Dataset(SBU_Dataset):
    def __getitem__(self, index):
        pose_values = self.loaded_videos[index][1]
        p1, p2 = torch.split(pose_values, [30,30], 1) # split 60 dim pose vector to individual partipants 30 dim pose
        avg_pose = torch.mean(torch.stack((p1,p2)), 0) # shape = (T,30) 
        y = torch.LongTensor([self.labels[index]])            
        return avg_pose, y




if __name__ == "__main__":

    from train_test_split import train_sets, test_sets
    import torchvision.transforms as transforms

    select_frame = 10
    res_size = 224

    transform = transforms.Compose([transforms.Resize([res_size, res_size]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    M1_train_set = M1_SBU_Dataset(train_sets[0], select_frame, mode='train', transform=transform, fold_no = 1)
    M1_valid_set = M1_SBU_Dataset(test_sets[0], select_frame, mode='valid', transform=transform, fold_no = 1)

    M2_train_set = M2_SBU_Dataset(train_sets[0], select_frame, mode='train', transform=transform, fold_no = 1)
    M2_valid_set = M2_SBU_Dataset(test_sets[0], select_frame, mode='valid', transform=transform, fold_no = 1)

    sample_X, sample_y = M1_train_set[4]
    print(f'M1 : X shape {sample_X.shape}')
    print(f'M1 : y shape {sample_y}')

    print()
    sample_X, sample_y = M2_train_set[4]
    print(f'M2 : X shape {sample_X.shape}')
    print(f'M2 : y shape {sample_y}')

    # print(f'first 10 train set folders : {train_set.folders[:10]}')
    # print(f'first 10 train set video lengths : {train_set.video_len[:10]}')
    # print(f'first 10 validation set folders : {valid_set.folders[:10]}')




