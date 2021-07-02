import os
import numpy as np
from sklearn.model_selection import KFold
import socket


if socket.gethostname() == 'beliveau':
    data_path = "/usr/local/data02/dpdataset/sbu"   # define sbu-kinect dataset path
elif socket.gethostname() == 'puck':
    data_path = "/usr/local/data01/rohitram/sbu"
else:
    raise Exception("invalid host!")

paths_to_sets = np.array(sorted([os.path.join(data_path,i) for i in os.listdir(data_path)])) # stores paths to the 21 participant sets e.g './sbu_dataset/s01s02'
train_sets = [] # contains 5 lists as a result of 5-fold CV where each list contains paths to sets denoting the training set
test_sets = [] # contains 5 lists as a result of 5-fold CV where each list contains paths to sets denoting the test set


kf = KFold(n_splits = 5, shuffle=True, random_state=0)
for train_index, test_index in kf.split(paths_to_sets):
    train_sets.append(paths_to_sets[train_index])
    test_sets.append(paths_to_sets[test_index])
    
with open(os.path.join(os.path.dirname(__file__), 'train_sets.txt'), 'w') as f:
    for fold_no, train_set in enumerate(train_sets,1):
        f.write(f'Fold no : {fold_no}\n')
        for path in train_set:
            f.write(f'{path}\n')
        f.write('\n')
            
with open(os.path.join(os.path.dirname(__file__), 'test_sets.txt'), 'w') as f:
    for fold_no, test_set in enumerate(test_sets,1):
        f.write(f'Fold no : {fold_no}\n')
        for path in test_set:
            f.write(f'{path}\n')
        f.write('\n')

