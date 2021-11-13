import pytorch_lightning as pl
from torch.utils.data import DataLoader

import numpy as np

import urllib
import os
import zipfile
from tqdm import tqdm
import shutil
import glob
from pathlib import Path
from sklearn.model_selection import train_test_split


from .datasets.hockey_dataset import M1_HockeyDataset, M2_HockeyDataset, M3_HockeyDataset, M4_HockeyDataset


class HockeyDataModule(pl.LightningDataModule):
    def __init__(self, CFG):
        super(HockeyDataModule, self).__init__()
        self.CFG = CFG
        self.data_dir = CFG.dataset_path

        if "resize" in CFG[CFG.training.model]:
            self.resize = CFG[CFG.training.model]["resize"]
        else:
            print(
                "NOTE : Frame features not used for this model. Resizing to 224x224."
            )
            self.resize = 224

        self.CATEGORY_TO_IDX = {"No_penalty":0, "Slashing":1, "Tripping":2}

    def prepare_data(self):
        pass

    def setup(self, fold_no=None):  # fold_no should be in [1,2,3,4,5]
        all_video_dirs = glob.glob(os.path.join(self.data_dir, '*', '*', 'FINAL_DATASET', '*'))
        penalty_idxs = [self.CATEGORY_TO_IDX[os.path.basename(Path(x).parents[1])] for x in all_video_dirs]
        X_train_dirs, X_test_dirs, y_train, y_test = train_test_split(all_video_dirs, penalty_idxs, test_size=0.2, random_state=42, stratify = penalty_idxs)

        if self.CFG.training.model == "model1":
            self.train_dataset = M1_HockeyDataset(
                (X_train_dirs, y_train),
                self.resize,
                "train",
                self.CFG["model1"].num_frames,
            )
            self.val_dataset = M1_HockeyDataset(
                (X_test_dirs, y_test),
                self.resize,
                "val",
                self.CFG["model1"].num_frames,
            )
        elif self.CFG.training.model == "model2":
            self.train_dataset = M2_HockeyDataset(
                (X_train_dirs, y_train),
                "train",
                self.CFG["model2"].num_frames,
                self.CFG["model2"].max_players,
                self.CFG["model2"].num_keypoints,
                self.CFG["model2"].coords_per_keypoint,
            )
            self.val_dataset = M2_HockeyDataset(
                (X_test_dirs, y_test),
                "val",
                self.CFG["model2"].num_frames,
                self.CFG["model2"].max_players,
                self.CFG["model2"].num_keypoints,
                self.CFG["model2"].coords_per_keypoint,
            )
        elif self.CFG.training.model == "model3":
            self.train_dataset = M3_HockeyDataset(
                (X_train_dirs, y_train),
                "train",
                self.CFG["model3"].num_frames,
                self.CFG["model3"].max_players,
                self.CFG["model3"].num_keypoints,
                self.CFG["model3"].coords_per_keypoint,
            )
            self.val_dataset = M3_HockeyDataset(
                (X_test_dirs, y_test),
                "val",
                self.CFG["model3"].num_frames,
                self.CFG["model3"].max_players,
                self.CFG["model3"].num_keypoints,
                self.CFG["model3"].coords_per_keypoint,
            ) 
        elif self.CFG.training.model == "model4":
            self.train_dataset = M4_HockeyDataset(
                (X_train_dirs, y_train),
                self.resize,
                "train",
                self.CFG.training.num_frames,
            )
            self.val_dataset = M4_HockeyDataset(
                (X_test_dirs, y_test),
                self.resize,
                "val",
                self.CFG.training.num_frames,
            )
        else:
            raise ValueError(f"invalid model name : {self.CFG.training.model}")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.CFG.training.train_dataloader)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.CFG.training.val_dataloader)

    def teardown(self,stage):
        import gc
        del self.train_dataset.loaded_videos
        del self.val_dataset.loaded_videos
        del self.train_dataset.folders
        del self.val_dataset.folders
        gc.collect()


if __name__ == "__main__":
    from omegaconf import OmegaConf

    CFG = OmegaConf.load("configs/hockey_dataset.yaml")

    CFG.dataset_path = '/usr/local/data01/rohitram/hockey_dataset'
    # print("\nModel 1 Test")
    # CFG.training.model = "model1"
    # dm = HockeyDataModule(CFG)
    # dm.prepare_data()
    # dm.setup()
    # for batch in dm.train_dataloader():
    #     print(f"frames = {batch[0].shape} , y = {batch[1].shape}")
    #     break

    # print("\nModel 2 Test")
    # CFG.training.model = "model2"
    # dm = HockeyDataModule(CFG)
    # dm.prepare_data()
    # dm.setup()
    # for batch in dm.train_dataloader():
    #     print(f"pose = {batch[0].shape} , y = {batch[1].shape}")
    #     break

    print("\nModel 3 Test")
    CFG.training.model = "model3"
    dm = HockeyDataModule(CFG)
    dm.prepare_data()
    dm.setup()
    for batch in dm.train_dataloader():
        print(f"pose = {batch[0].shape} , mask = {batch[1].shape}, y = {batch[2].shape}")
        break

    # print("\nModel 4 Test")
    # CFG.training.model = "model4"
    # dm = HockeyDataModule(CFG)
    # dm.prepare_data()
    # dm.setup(fold_no=1)
    # for batch in dm.train_dataloader():
    #     print(
    #         f"frames = {batch[0].shape} , pose = {batch[1].shape}, y = {batch[2].shape}"
    #     )
    #     break

