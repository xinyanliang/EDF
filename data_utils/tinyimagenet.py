#encoding=utf-8
#author: liang xinyan
#email: liangxinyan48@163.com
import numpy as np
import shutil
import os
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool
opt = os.path
from config import get_configs
paras = get_configs()
image_size = paras['image_size']
data_name = paras['data_name']
w, h, c = image_size['w'], image_size['h'], image_size['c']

import sklearn


def read_image(fn):
    img = Image.open(fn)
    img = img.convert('L')
    # img = img.convert('RGB')
    img = img.resize((w, h))
    img = np.array(img)
    return np.array(img)

with open(opt.join(data_name, 'wnids.txt')) as f:
    labels = f.readlines()
    labels = [str.strip(i) for i in labels]

# os.makedirs(data_name, exist_ok=True)
train_X_npy = opt.join(data_name, 'train_X.npy')
train_Y_npy = opt.join(data_name, 'train_Y.npy')
test_X_npy = opt.join(data_name, 'test_X.npy')
test_Y_npy = opt.join(data_name, 'test_Y.npy')

def save_train():
    # tiny-imagenet-200\train\n01443537\images
    train_x, train_y = [], []
    for idx, label in enumerate(labels):
        each_class_dir = opt.join(data_name, 'train', label, 'images')
        img_names = os.listdir(each_class_dir)
        for img_name in img_names:
            train_x.append(read_image(opt.join(each_class_dir, img_name)))
            train_y.append(idx)
    train_y = np.array(train_y)
    train_x = np.array(train_x)
    print(train_x.shape)
    np.save(train_X_npy, train_x)
    # np.save(train_Y_npy, train_y)

# save_train()
def save_test():
    # tiny-imagenet200\val\images
    test_x, test_y = [], []
    with open(opt.join(data_name, 'val', 'val_annotations.txt')) as f:
        # val_0.JPEG	n03444034
        for i in f.readlines():
            ii = i.split('\t')
            # print(ii[0], ii[1])
            test_x.append(read_image(opt.join(data_name, 'val', 'images', ii[0])))
            test_y.append(labels.index(ii[1]))

    test_x = np.array(test_x)
    test_y = np.array(test_y)
    print(test_x.shape)
    np.save(test_X_npy, test_x)
    # np.save(test_Y_npy, test_y)
save_test()
save_train()








