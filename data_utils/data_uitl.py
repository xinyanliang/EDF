#encoding=utf-8
#author: liang xinyan
#email: liangxinyan48@163.com
import tensorflow as tf
import numpy as np
import os
import config
opt = os.path
paras = config.get_configs()
nb_view = paras['nb_view']
classes = paras['classes']
image_size = paras['image_size']
w, h, c = image_size['w'], image_size['h'], image_size['c']

def get_data(data_base_dir='..'):
    print('Data loading ......')
    train_x = np.load(os.path.join(data_base_dir, 'train_X.npy'))
    test_x = np.load(os.path.join(data_base_dir, 'test_X.npy'))
    if c == 1:
        train_x = np.expand_dims(train_x, axis=-1)
        test_x = np.expand_dims(test_x, axis=-1)
    train_x = (train_x / 127.5) - 1.
    test_x = (test_x / 127.5) - 1.
    train_y = np.load(os.path.join(data_base_dir, 'train_Y.npy'))
    test_y = np.load(os.path.join(data_base_dir, 'test_Y.npy'))
    train_y = tf.keras.utils.to_categorical(train_y)
    test_y = tf.keras.utils.to_categorical(test_y)
    print('Data loading finished！！！')
    return train_x, train_y, test_x, test_y


def get_views(data_name='AWA1', view_data_dir='view', idx_split=0):
    from . import AWA1, Reuter, mfeat, nus_wide
    if data_name == 'AWA1':
        view_train_x, train_y, view_test_x, test_y = AWA1.load_AWA1(
            view_data_dir=view_data_dir, idx_split=idx_split)
    if data_name == 'Reuters':
        view_train_x, train_y, view_test_x, test_y = Reuter.load_Reuter(
            view_data_dir=view_data_dir, idx_split=idx_split)
    if data_name == 'mfeat':
        view_train_x, train_y, view_test_x, test_y = mfeat.load_mfeat(
            view_data_dir=view_data_dir, idx_split=idx_split)
    if data_name == 'nus_wide':
        view_train_x, train_y, view_test_x, test_y = nus_wide.load_nus_wide(
            view_data_dir=view_data_dir, idx_split=idx_split)

    if data_name in ['ChemBook',  'Chembl', 'PubChem', 'tiny-imagenet200']:
        models_ls = ['resnet50', 'desnet121', 'MobileNetV2', 'Xception', 'InceptionV3']
        if nb_view == 10:
            models_ls = models_ls+['resnet18', 'resnet34', 'desnet169', 'desnet201', 'NASNetMobile']
        view_train_x = []
        view_test_x = []
        for model in models_ls:
            view_train_x.append(np.load(os.path.join(view_data_dir, model+'train_X.npy')))
            view_test_x.append(np.load(os.path.join(view_data_dir, model+'test_X.npy')))
        train_y = np.load(os.path.join(view_data_dir, 'train_Y.npy'))
        test_y = np.load(os.path.join(view_data_dir, 'test_Y.npy'))
    train_y = tf.keras.utils.to_categorical(train_y, classes)
    test_y = tf.keras.utils.to_categorical(test_y, classes)
    # import keras
    # keras.utils.to_categorical()

    return view_train_x, train_y, view_test_x, test_y

def preprocess_input(data_saved_dir='database', save_name='x'):
    '''
    Preprocess tha data to ensure model to be able to it
    :param save_dir: path of data to save
    :param save_name: name of data
    :return: preprocessed data
    '''
    x = np.load(os.path.join(data_saved_dir, save_name+'.npy'))
    if len(x.shape) == 3:
        x = np.expand_dims(x, axis=-1)
    x = (x / 127.5) - 1.
    print('Data loading finished！！！')
    return x

if __name__ == '__main__':
    base_dir = opt.join('fn')
    # train_fns, train_y, test_fns, test_y = get_image_paths(base_dir=base_dir)
    # train_fns = [opt.join('data', v) for v in train_fns]
    # train_fns = [v.split('_')[0] for v in train_fns]
    # print(len(set(train_fns)))
    # print(train_fns)
