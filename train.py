import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.utils.data as data
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
# from sbu_functions import *
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
import pickle
from collections import defaultdict, OrderedDict
import glob
import sys

data_path = "/usr/local/data02/dpdataset/sbu"   # define sbu-kinect dataset path

paths_to_sets = np.array([os.path.join(data_path,i) for i in os.listdir(data_path)]) # stores paths to the 21 participant sets e.g './sbu_dataset/s01s02'
train_sets = [] # contains 5 lists as a result of 5-fold CV where each list contains paths to sets denoting the training set
test_sets = [] # contains 5 lists as a result of 5-fold CV where each list contains paths to sets denoting the test set


kf = KFold(n_splits = 5)
for train_index, test_index in kf.split(paths_to_sets):
    train_sets.append(paths_to_sets[train_index])
    test_sets.append(paths_to_sets[test_index])
    # print(test_index)


def get_train_test_data(train_set, test_set):
    """
    Args
        train_set : list of paths to participant categories (e.g '../../s0102') for train set
        test_set :  list of paths to participant categories (e.g '../../s0102')for test set

    Returns
        tuple (train_data, test_data).

        train_data : List[ tuple(video_path, label, num_frames) ].
        test_data : List[ tuple(video_path, label, num_frames) ].

        Each path in 'train_set' and 'test_set' arguments are expanded respectively to generate a list of all paths to videos contained
        in them and the number of frames in each video. Label is category of video (0-7).

    """
    
    train_data = []
    test_data = []

    for part_path in train_set: 
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
                train_data.append((run_path,label,num_frames))

    for part_path in test_set: 
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
                test_data.append((run_path,label,num_frames))

    return train_data, test_data


train_data, test_data = get_train_test_data(train_sets[0], test_sets[0]) # using only first fold here