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

from sklearn import preprocessing


def read_image(fn):
    img = Image.open(fn)
    # img = img.convert('L')
    img = img.convert('RGB')
    img = img.resize((w, h))
    img = np.array(img)
    return np.array(img)


# os.makedirs(data_name, exist_ok=True)
train_X_npy = opt.join(data_name, 'train_X.npy')
train_Y_npy = opt.join(data_name, 'train_Y.npy')
test_X_npy = opt.join(data_name, 'test_X.npy')
test_Y_npy = opt.join(data_name, 'test_Y.npy')

def save_dog():
    # /export/lxy/TEC/dogs/train/n02115913-dhole
    def read(data_type='train'):
        x, y = [], []
        train_dir = opt.join(data_name, data_type)
        for label in os.listdir(train_dir):
            each_label_dir = opt.join(train_dir, label)
            for img_name in os.listdir(each_label_dir):
                x.append(read_image(opt.join(each_label_dir, img_name)))
                y.append(label)
        return x, y

    train_x, train_y = read(data_type='train')
    train_x = np.array(train_x)
    np.save(train_X_npy, train_x)
    print(train_x.shape)
    LB = preprocessing.LabelEncoder()
    LB.fit(train_y)
    train_y = LB.transform(train_y)
    np.save(train_Y_npy, train_y)

    test_x, test_y = read(data_type='test')
    test_x = np.array(test_x)
    test_y = LB.transform(test_y)
    np.save(test_X_npy, test_x)
    np.save(test_Y_npy, test_y)
    print(test_x.shape)

save_dog()








