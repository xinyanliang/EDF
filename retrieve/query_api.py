#encoding=utf-8
#author: liang xinyan
#email: liangxinyan48@163.com
import os
import numpy as np
from data_utils.npy_util import read_image
from features import feature
from retrieve import retrieve

database_vecs = np.load(os.path.join('database', 'database.npy'))
database_vecs = np.load(os.path.join('database', 'x.npy'))

edf_model_name = '3-2-0-1-0-4-0'

def query(query_img_url):
    query_img = read_image(query_img_url)
    query_img = np.expand_dims(query_img, axis=0)
    query_img = np.expand_dims(query_img, axis=-1)
    query_img = (query_img / 127.5) - 1.
    view_models = ['resnet50', 'desnet121', 'MobileNetV2', 'Xception', 'InceptionV3']
    Feats = feature.Feature()
    views = Feats.get_feats_multi_views(view_models, x=query_img, save_data_suffix=None)
    x_feats = Feats.get_feats_by_edf(views=views, save_data_suffix=None, edf_model_name=edf_model_name)
    topk_imgs = retrieve.topk_imgs(query_img=x_feats,
                                   database_vecs=database_vecs,
                                   database_imgs=database_imgs,
                                   topk=10,
                                   metric="euclidean")
    return topk_imgs

