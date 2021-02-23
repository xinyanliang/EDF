#encoding=utf-8
#author: liang xinyan
#email: liangxinyan48@163.com
import numpy as np
import tensorflow as tf
import os
from sklearn.metrics import pairwise_distances

a = ['PubChem', 'Chembl', 'ChemBook']
os.environ["CUDA_VISIBLE_DEVICES"] = '6'
model_data = a[2]
data_name  = a[1]
data_name = model_data+'-'+data_name
print(data_name)
data_base_dir = os.path.join('..', data_name)
data_dir = os.path.join(data_base_dir, data_name+'-EDF')
if model_data == 'PubChem':
    code = '3-2-0-1-0-4-0'  # PubChem
if model_data == 'Chembl':
    code = '4-0-3-1-2-3-4-0-4'  # Chembl
if model_data == 'ChemBook':
    code = '0-1-4-2-2-1-1'  # ChemBook ef10


def get_data():
    train_x = np.load(os.path.join(data_dir, code+'train_X.npy'))
    train_y = np.load(os.path.join(data_base_dir, 'train_Y.npy'))
    test_x = np.load(os.path.join(data_dir, code+'test_X.npy'))
    test_y = np.load(os.path.join(data_base_dir, 'test_Y.npy'))
    return train_x, train_y, test_x, test_y


def cal_dist(topk=[1, 5, 10], metric="euclidean"):
    train_x, train_y, test_x, test_y = get_data()
    num_test = test_x.shape[0]
    train_y = np.tile(train_y, (num_test, 1))
    dis = pairwise_distances(X=test_x, Y=train_x, metric=metric, n_jobs=-1)
    sort_idx1 = np.argsort(dis, axis=1)
    def report_topk(k):
        sort_idx = sort_idx1[:, :k]
        count = 0
        for i in range(num_test):
            if test_y[i] in train_y[i, sort_idx[i, :]]:
                count += 1
        print(count/num_test)
    for k in topk:
        report_topk(k)


'''
Step 1: get_feats.py
Step 2: get_feats_by_EDF.py
Step 3: retrieve.py
'''
cal_dist(topk=[1, 5, 10, 15, 20, 50], metric="euclidean")
# cal_dist(topk=[1, 5, 10, 15, 20, 50], metric="cosine")




