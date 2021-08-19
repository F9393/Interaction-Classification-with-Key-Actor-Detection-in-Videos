import pytorch_lightning as pl
from torch.utils.data import DataLoader

import numpy as np

import urllib
import os
import zipfile
from tqdm import tqdm
import shutil
import glob


from .sbu_dataset import M1_SBU_Dataset, M2_SBU_Dataset, M3_SBU_Dataset, M4_SBU_Dataset


class SBUDataModule(pl.LightningDataModule):
    def __init__(self, CFG):
        super(SBUDataModule, self).__init__()
        self.CFG = CFG
        self.data_dir = CFG.dataset_path

        self.sets = [
            "s01s02",
            "s01s03",
            "s01s07",
            "s02s01",
            "s02s03",
            "s02s06",
            "s02s07",
            "s03s02",
            "s03s04",
            "s03s05",
            "s03s06",
            "s04s02",
            "s04s03",
            "s04s06",
            "s05s02",
            "s05s03",
            "s06s02",
            "s06s03",
            "s06s04",
            "s07s01",
            "s07s03",
        ]

        self.folds = [
            [1, 9, 15, 19],
            [5, 7, 10, 16],
            [2, 3, 20, 21],
            [4, 6, 8, 11],
            [12, 13, 14, 17, 18],
        ]

        if "resize" in CFG[CFG.training.model]:
            self.resize = CFG[CFG.training.model]["resize"]
        elif CFG.cache_folds:
            print(
                "NOTE : Although frame features are not used for this model, folds will be cached after resizing images to 224x224. Delete cache or set 'use_cache' in config to false and run again if you want to resize to another dimension."
            )
            self.resize = 224
        else:
            self.resize = 224

        self.select_frames = 10

    def prepare_data(self):
        if os.path.exists(self.data_dir):
            data_files = os.listdir(self.data_dir)
            if set(self.sets).issubset(set(data_files)):
                return
            if set([i for i in range(1, 22)]).issubset(set(data_files)):
                print("removing extra directory (1,2...21)")
                for i in range(1, 22):
                    number_path = os.path.join(self.data_dir, str(i))
                    shutil.move(glob.glob(f"{number_path}{os.sep}s*")[0], number_path)
                    shutil.rmtree(number_path)
                return

        print(f"downloading SBU Kinect dataset into {os.path.abspath(self.data_dir)}")

        sbu_files = [
            "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s01s02.zip",
            "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s01s03.zip",
            "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s01s07.zip",
            "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s02s01.zip",
            "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s02s03.zip",
            "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s02s06.zip",
            "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s02s07.zip",
            "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s03s02.zip",
            "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s03s04.zip",
            "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s03s05.zip",
            "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s03s06.zip",
            "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s04s02.zip",
            "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s04s03.zip",
            "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s04s06.zip",
            "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s05s02.zip",
            "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s05s03.zip",
            "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s06s02.zip",
            "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s06s03.zip",
            "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s06s04.zip",
            "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s07s01.zip",
            "http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/s07s03.zip",
        ]

        os.makedirs(self.data_dir, exist_ok=True)

        with tqdm(range(1, 22)) as pbar:
            for i in pbar:
                pbar.set_description(
                    f"downloading set {os.path.basename(sbu_files[i-1])}"
                )
                urllib.request.urlretrieve(
                    sbu_files[i - 1], f"./sbu_dataset/Set_{i}.zip"
                )

        with tqdm(range(1, 22)) as pbar:
            for i in pbar:
                filename = f"Set_{i}"
                pbar.set_description(f"unzipping {filename}.zip")
                zipped_path = os.path.join(self.data_dir, filename + ".zip")
                file = zipfile.ZipFile(zipped_path, "r")
                unzip_path = os.path.join(self.data_dir, filename)
                file.extractall(unzip_path)

        with tqdm(range(1, 22)) as pbar:
            for i in pbar:
                pbar.set_description("removing unnecessary files")
                target_path = os.path.join(self.data_dir, f"Set_{i}")
                shutil.rmtree(os.path.join(target_path, "__MACOSX"))
                shutil.move(glob.glob(f"{target_path}{os.sep}s*")[0], self.data_dir)
                shutil.rmtree(target_path)

    def setup(self, fold_no):  # fold_no should be in [1,2,3,4,5]

        train_sets = []
        val_sets = []

        all_sets = np.array(self.sets)
        for fold in self.folds:
            ind = np.array(fold) - 1
            val_sets.append(np.core.defchararray.add(self.data_dir.rstrip(os.sep) + os.sep, all_sets[ind]))
            train_sets.append(
                np.core.defchararray.add(self.data_dir.rstrip(os.sep) + os.sep, np.delete(all_sets, ind))
            )
            assert (
                len(train_sets[-1]) + len(val_sets[-1]) == 21
            ), "train test split incorrect!"

        reqd_train_set_paths = train_sets[fold_no - 1]
        reqd_val_set_paths = val_sets[fold_no - 1]

        if self.CFG.training.model == "model1":
            self.train_dataset = M1_SBU_Dataset(
                reqd_train_set_paths,
                self.select_frames,
                "train",
                self.resize,
                fold_no,
                self.data_dir,
                self.CFG.cache_folds,
                self.CFG.use_cache,
            )
            self.val_dataset = M1_SBU_Dataset(
                reqd_val_set_paths,
                self.select_frames,
                "val",
                self.resize,
                fold_no,
                self.data_dir,
                self.CFG.cache_folds,
                self.CFG.use_cache,
            )
        elif self.CFG.training.model == "model2":
            self.train_dataset = M2_SBU_Dataset(
                self.CFG["model2"].pose_coord,
                reqd_train_set_paths,
                self.select_frames,
                "train",
                self.resize,
                fold_no,
                self.data_dir,
                self.CFG.cache_folds,
                self.CFG.use_cache,
            )
            self.val_dataset = M2_SBU_Dataset(
                self.CFG["model2"].pose_coord,
                reqd_val_set_paths,
                self.select_frames,
                "val",
                self.resize,
                fold_no,
                self.data_dir,
                self.CFG.cache_folds,
                self.CFG.use_cache,
            )
        elif self.CFG.training.model == "model3":
            self.train_dataset = M3_SBU_Dataset(
                self.CFG["model3"].pose_coord,
                reqd_train_set_paths,
                self.select_frames,
                "train",
                self.resize,
                fold_no,
                self.data_dir,
                self.CFG.cache_folds,
                self.CFG.use_cache,
            )
            self.val_dataset = M3_SBU_Dataset(
                self.CFG["model3"].pose_coord,
                reqd_val_set_paths,
                self.select_frames,
                "val",
                self.resize,
                fold_no,
                self.data_dir,
                self.CFG.cache_folds,
                self.CFG.use_cache,
            )
        elif self.CFG.training.model == "model4":
            self.train_dataset = M4_SBU_Dataset(
                self.CFG["model4"].pose_coord,
                reqd_train_set_paths,
                self.select_frames,
                "train",
                self.resize,
                fold_no,
                self.data_dir,
                self.CFG.cache_folds,
                self.CFG.use_cache,
            )
            self.val_dataset = M4_SBU_Dataset(
                self.CFG["model4"].pose_coord,
                reqd_val_set_paths,
                self.select_frames,
                "val",
                self.resize,
                fold_no,
                self.data_dir,
                self.CFG.cache_folds,
                self.CFG.use_cache,
            )
        else:
            raise ValueError(f"invalid model name : {self.CFG.training.model}")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.CFG.training.train_dataloader)

    def val_dataloader(self):
        return DataLoader(self.train_dataset, **self.CFG.training.val_dataloader)


if __name__ == "__main__":
    import hydra

    hydra.initialize(config_path="../../configs")
    DEF_CFG = hydra.compose(config_name="config")
    CFG = DEF_CFG.dataset
    dm = SBUDataModule(CFG)
    dm.prepare_data()
    dm.setup(fold_no=1)