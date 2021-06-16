import os
import numpy as np
from sklearn.model_selection import KFold

data_path = "/usr/local/data02/dpdataset/sbu"   # define sbu-kinect dataset path

paths_to_sets = np.array([os.path.join(data_path,i) for i in os.listdir(data_path)]) # stores paths to the 21 participant sets e.g './sbu_dataset/s01s02'
train_sets = [] # contains 5 lists as a result of 5-fold CV where each list contains paths to sets denoting the training set
test_sets = [] # contains 5 lists as a result of 5-fold CV where each list contains paths to sets denoting the test set


kf = KFold(n_splits = 5)
for train_index, test_index in kf.split(paths_to_sets):
    train_sets.append(paths_to_sets[train_index])
    test_sets.append(paths_to_sets[test_index])