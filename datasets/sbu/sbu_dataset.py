import os
from PIL import Image
import torch
from torch.utils import data
from tqdm import tqdm
import glob
import pickle

class SBU_Dataset(data.Dataset):
    def __init__(self, set_paths, select_frames, mode, transform=None):
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
        self.transform = transform
        self.mode = mode
        self.loaded_videos = None
    
        if mode == 'train':
            loaded_dataset_path = os.path.join(os.path.dirname(__file__), 'sbu_train.pkl')
            if os.path.exists(loaded_dataset_path):
                with open(loaded_dataset_path, 'rb') as f:
                    self.loaded_videos = pickle.load(f)
        if mode == 'valid':
            loaded_dataset_path = os.path.join(os.path.dirname(__file__), 'sbu_valid.pkl')
            if os.path.exists(loaded_dataset_path):
                with open(loaded_dataset_path, 'rb') as f:
                    self.loaded_videos = pickle.load(f)
        if self.loaded_videos is None:
            self.loaded_videos = self.load_videos()

    def get_video_data(self, set_paths):
        """
        Args
            set_paths : list of paths to participant categories (e.g '../../s0102') for corresponding set(train/test)

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

        loaded_videos = []
        print(f'Loading {self.mode} Data!')
        for index, video_pth in enumerate(tqdm(self.folders)):
            video_len = self.video_len[index]
            start_idx = (video_len - self.select_frames) // 2
            end_idx = start_idx + self.select_frames

            frame_pths = glob.glob(f'{video_pth}/rgb*')[start_idx : end_idx]
            loaded_imgs = []
            for frame_pth in frame_pths:
                frame = Image.open(frame_pth).convert('RGB')
                frame = self.transform(frame) if self.transform is not None else frame  # impose transformation if exists
                loaded_imgs.append(frame.squeeze_(0))
            loaded_imgs = torch.stack(loaded_imgs, dim=0)
            loaded_videos.append(loaded_imgs)

        assert len(loaded_videos) == len(self.folders), "error in reading images of videos"
        return loaded_videos

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, index):
        X = self.loaded_videos[index]   
        y = torch.LongTensor([self.labels[index]])            
        return X, y




if __name__ == "__main__":

    from train_test_split import train_sets, test_sets
    import torchvision.transforms as transforms

    select_frame = 10
    res_size = 224

    transform = transforms.Compose([transforms.Resize([res_size, res_size]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_set = SBU_Dataset(train_sets[0], select_frame, mode='train', transform=transform)
    valid_set = SBU_Dataset(test_sets[0], select_frame, mode='valid', transform=transform)

    sample_X, sample_y = train_set[4]

    print(f'X shape {sample_X.shape}')
    print(f'y shape {sample_y}')

    print(f'first 10 train set folders : {train_set.folders[:10]}')
    print(f'first 10 validation set folders : {valid_set.folders[:10]}')




