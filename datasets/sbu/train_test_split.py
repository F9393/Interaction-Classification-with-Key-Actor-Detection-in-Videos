import os
import numpy as np
import socket

from .sbu_dataset import folds_cache_path

if socket.gethostname() == 'beliveau':
    data_path = "/usr/local/data02/dpdataset/sbu/"   # define sbu-kinect dataset path
elif socket.gethostname() == 'puck':
    data_path = "/usr/local/data01/rohitram/sbu/"
else:
    raise Exception("invalid host!")

SETS = np.array(['s01s02','s01s03','s01s07','s02s01','s02s03','s02s06','s02s07','s03s02',
        's03s04','s03s05','s03s06','s04s02','s04s03','s04s06','s05s02','s05s03',
        's06s02','s06s03','s06s04','s07s01','s07s03'])

FOLDS = [[ 1,  9, 15, 19],
        [ 5,  7, 10, 16],
        [ 2,  3, 20, 21],
        [ 4,  6,  8, 11],
        [12, 13, 14, 17, 18]]

train_sets = []
test_sets = []

for fold in FOLDS:
    ind = np.array(fold) - 1
    test_sets.append(np.core.defchararray.add(data_path, SETS[ind]))
    train_sets.append(np.core.defchararray.add(data_path, np.delete(SETS,ind)))
    assert len(train_sets[-1]) + len(test_sets[-1]) == 21, "train test split incorrect!"

    
if not os.path.exists(os.path.join(folds_cache_path, 'train_sets.txt')):    
    with open(os.path.join(folds_cache_path, 'train_sets.txt'), 'w') as f:
        for fold_no, train_set in enumerate(train_sets,1):
            f.write(f'Fold no : {fold_no}\n')
            for path in train_set:
                f.write(f'{path}\n')
            f.write('\n')

if not os.path.exists(os.path.join(folds_cache_path, 'test_sets.txt')):  
    with open(os.path.join(folds_cache_path, 'test_sets.txt'), 'w') as f:
        for fold_no, test_set in enumerate(test_sets,1):
            f.write(f'Fold no : {fold_no}\n')
            for path in test_set:
                f.write(f'{path}\n')
            f.write('\n')

if __name__ == "__main__":
    print(train_sets)
    print(test_sets)   