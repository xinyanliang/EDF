#encoding=utf-8
#author: liang xinyan
#email: liangxinyan48@163.com
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '7'
import tensorflow as tf
import utils
import time
import numpy as np
# https://stackoverflow.com/questions/60130622/warningtensorflow-with-constraint-is-deprecated-and-will-be-removed-in-a-future
tf.get_logger().setLevel('ERROR')
from data_utils.data_uitl import get_views, get_data
import config
paras = config.get_configs()
data_name = paras['data_name']

view_train_x, train_y, view_test_x, test_y = get_views(
    view_data_dir=os.path.join('..', data_name, 'view'))
def metrics(test_y, y_pred):
    topk1 = tf.keras.metrics.top_k_categorical_accuracy(test_y, y_pred, k=1)
    topk5 = tf.keras.metrics.top_k_categorical_accuracy(test_y, y_pred, k=5)
    topk10 = tf.keras.metrics.top_k_categorical_accuracy(test_y, y_pred, k=10)

    # with tf.Session() as sess:
    topk1 = topk1.numpy()
    topk5 = topk5.numpy()
    topk10 = topk10.numpy()
    topk1 = topk1[topk1 == 1].shape[0] / 10000.
    topk5 = topk5[topk5 == 1].shape[0] / 10000.
    topk10 = topk10[topk10 == 1].shape[0] / 10000.
    print(topk1, topk5, topk10)

def sign_sqrt(x):
    # mfb_sign_sqrt = torch.sqrt(F.relu(mfb_out)) - torch.sqrt(F.relu(-mfb_out))
    return tf.keras.backend.sign(x) * tf.keras.backend.sqrt(tf.keras.backend.abs(x) + 1e-10)

def l2_norm(x):
    return tf.keras.backend.l2_normalize(x, axis=-1)

def multi_view_result_summary(base_dir='Chembl_view_result/EDF-True-64-5result', code='2-0-3-1-2-0-0'):
    def sign_sqrt(x):
        # mfb_sign_sqrt = torch.sqrt(F.relu(mfb_out)) - torch.sqrt(F.relu(-mfb_out))
        return tf.keras.backend.sign(x) * tf.keras.backend.sqrt(tf.keras.backend.abs(x) + 1e-10)

    def l2_norm(x):
        return tf.keras.backend.l2_normalize(x, axis=-1)

    # _custom_objects = {"sign_norm": sign_norm, 'tf': tf}
    # https://www.jianshu.com/p/dd2bebfa867d
    _custom_objects = {"sign_sqrt": sign_sqrt, 'l2_norm': l2_norm, 'tf': tf}
    model = tf.keras.models.load_model(base_dir+'/'+code+'.h5', custom_objects=_custom_objects)
    print(model.summary())
    nb_views = utils.get_nb_view_by_individal_code(code)
    code_list = [int(i) for i in code.strip().split('-')]
    views = code_list[:nb_views]
    view_test_xx = [view_test_x[i] for i in views]
    start = time.time()
    y_pred = model.predict(view_test_xx)
    stop = time.time()
    interval = stop-start
    print(interval, 10000/interval, interval/10000)
    metrics(test_y, y_pred)


def single_view_result_summary(model_name, base_dir=os.path.join('Chemblresult', 'best')):
    model = tf.keras.models.load_model(base_dir+'/'+model_name)
    print(model.summary())
    # y_pred = model.predict(view_test_x)
    # metrics(test_y, y_pred)


train_xx, train_yy, test_xx, test_yy = get_data(data_base_dir=os.path.join('..', data_name))

def original(model_name, base_dir=os.path.join(data_name+'result', 'best')):
    model = tf.keras.models.load_model(base_dir+'/' + model_name+'.h5')
    # print(model.summary())
    print(model_name)
    start = time.time()
    y_pred = model.predict(test_xx)
    cost = time.time()-start
    print(cost, cost/10000, 10000/cost)
    metrics(test_yy, y_pred)


def ensemble(models_ls):
    def single(model_name, base_dir=os.path.join(data_name + 'result', 'best')):
        model = tf.keras.models.load_model(base_dir + '/' + model_name + '.h5')
        start = time.time()
        y_pred = model.predict(test_xx)
        cost = time.time() - start
        return y_pred, cost

    y_preds, y, costs = [], [], []
    for model_name in models_ls:
        print(model_name)
        y_pred, cost = single(model_name=model_name)
        y_preds.append(y_pred)
        costs.append(cost)
    y_preds = np.array(y_preds)
    costs = np.array(costs)
    y_pred_avg = np.mean(y_preds, axis=0)
    y_pred_max = np.max(y_preds, axis=0)
    metrics(test_yy, y_pred_avg)
    metrics(test_yy, y_pred_max)
    costs = np.sum(costs)
    print('Time', costs, costs / 10000, 10000/costs)




if __name__ == '__main__':
    # code = [str(i) for i in [0, 6, 6, 3, 2, 8, 9, 7, 6, 4, 7, 1, 3, 8, 4, 7, 1, 2, 5, 2, 2, 4, 2, 2, 3, 4, 4, 0, 4, 0, 4, 0, 0, 4, 0, 2, 4]]
    # code = '-'.join(code)
    # multi_view_result_summary(base_dir='EF29_result', code=code)
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    # for i in ['add', 'mul', 'cat', 'max', 'avg']:
    #     print(i)
    #     single_view_result_summary(base_dir='view_result-128', model_name=i+'.h5')
    print(data_name)
    models_ls = ['resnet50', 'desnet121', 'MobileNetV2', 'Xception', 'InceptionV3']
    # models_ls += ['resnet18', 'resnet34',  'desnet169', 'desnet201', 'NASNetMobile']
    # models_ls = ['MobileNetV2']
    # ensemble(models_ls)
    # for i in models_ls:
    #     original(model_name=i)
    # original(model_name='')

    # base_dir = os.path.join(data_name+'_view_result', 'EDF', 'EDF-True-256-10result')
    base_dir = os.path.join(data_name+'_view_result', 'EDF-Ture-128-5result')
    # base_dir = os.path.join('..', 'TEC1', 'EF', 'EF2_result')
    code = '3-2-3-4-1-3-4-1-2-0-1-2-2-1-2-3-1-1-3-0-3-3-0-3-0-0-0-3-0-4-4-3-4-3-0-4-4'
    multi_view_result_summary(base_dir=base_dir, code=code)
    a = code
    b = [int(i) for i in a.split('-')]
    b_len = len(b) // 2 + 1
    print(b_len)
    print(b[:b_len])
    print(b[b_len:])