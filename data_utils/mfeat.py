#encoding=utf-8
#author: liang xinyan
#email: liangxinyan48@163.com
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import os


def split_train_test(x, y, n_splits=5, test_size=0.2, seed=1024):
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
    train_idxs, test_idxs = [], []
    for train_idx, test_idx in sss.split(x, y):
        train_idxs.append(train_idx)
        test_idxs.append(test_idx)
    return train_idxs, test_idxs

def load_mfeat(view_data_dir, n_splits=5, idx_split=0, test_size=0.2, seed=1024):
    print('********************** idx_split:', idx_split)
    view_names = ['mfeat-fac', 'mfeat-fou', 'mfeat-kar', 'mfeat-mor', 'mfeat-pix', 'mfeat-zer']
    x = np.load(os.path.join(view_data_dir, view_names[-1] + '.npy'))
    y = np.load(os.path.join(view_data_dir, 'y.npy'))

    train_idxs, test_idxs = split_train_test(x=x, y=y, n_splits=n_splits,
                                            test_size=test_size, seed=seed)
    view_train_x, view_test_x = [], []
    for view_name in view_names:
        x = np.load(os.path.join(view_data_dir, view_name+'.npy'))
        view_train_x.append(x[train_idxs[idx_split]])
        view_test_x.append(x[test_idxs[idx_split]])
    train_y = y[train_idxs[idx_split]]
    test_y = y[test_idxs[idx_split]]
    return view_train_x, train_y, view_test_x, test_y
