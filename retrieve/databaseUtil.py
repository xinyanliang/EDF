#encoding=utf-8
#author: liang xinyan
#email: liangxinyan48@163.com

from features import feature
from data_utils import npy_util
import os
import numpy as np
from data_utils import data_uitl


def construct_retrieve_database_test(edf_model_name='3-2-0-1-0-4-0'):
    '''
    EDF and view exteacters are trained on PubChem-10k dataset;
    retrieve database is constructed using training set of ChEMBL-10k dataset;
    these images from test set of ChEMBL-10k dataset are used query images.
    :param edf_model_name:
    :return:
    '''
    train_x, train_y, test_x, test_y = data_uitl.get_data('database')
    x = [train_x, test_x]
    view_models = ['resnet50', 'desnet121', 'MobileNetV2', 'Xception', 'InceptionV3']
    save_data_suffix = ['train_X', 'test_X']
    Feats = feature.Feature()
    for i in range(len(x)):
        views = Feats.get_feats_multi_views(view_models, x=x[i], save_data_suffix=save_data_suffix[i])
        Feats.get_feats_by_edf(views=views, save_data_suffix=save_data_suffix[i], edf_model_name=edf_model_name)

def imgs2npy(imgs_file_list, save_dir='database', save_name='x'):
    '''
    Read images according to their path, and then save them in the format of npy
    :param imgs_file_list: path of images to read
    :param save_name: path of npy file to save
    :return: images in the format of array of numpy
    '''
    imgs = []
    for img_fn in imgs_file_list:
        imgs.append(npy_util.read_image(img_fn))
    imgs = np.array(imgs)
    np.save(os.path.join(save_dir, save_name), imgs)
    return imgs


def construct_retrieve_database():
    x = data_uitl.preprocess_input(data_saved_dir='database', save_name='x')
    view_models = ['resnet50', 'desnet121', 'MobileNetV2', 'Xception', 'InceptionV3']
    Feats = feature.Feature(model_dir='models', save_data_dir='database', database_name='database')
    views = Feats.get_feats_multi_views(view_models, x=x, save_data_suffix=None)
    Feats.get_feats_by_edf(views=views, save_data_suffix=None, edf_model_name='3-2-0-1-0-4-0')

