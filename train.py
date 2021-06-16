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
from sklearn.metrics import accuracy_score
from collections import defaultdict, OrderedDict
import glob
import sys

from datasets.sbu.train_test_split import train_sets, test_sets # generates K-fold train and test sets
from datasets.sbu.sbu_dataset import  SBU_Dataset



# EncoderCNN architecture
res_size = 224        # ResNet image size

# DecoderRNN architecture
# to-do

# training parameters
k = 8             # number of target category
epochs = 50        # training epochs
batch_size = 16
learning_rate = 1e-3
lr_patience = 15
log_interval = 10   # interval for displaying training info

# Select given number of middle frames (left-biased). For sbu-dataset this has to be <=10. 
select_frame = 10


transform = transforms.Compose([transforms.Resize([res_size, res_size]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


# using first fold
train_set = SBU_Dataset(train_sets[0], select_frame, mode='train', transform=transform)
valid_set = SBU_Dataset(test_sets[0], select_frame, mode='valid', transform=transform)

