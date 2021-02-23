#encoding=utf-8
#author: liang xinyan
#email: liangxinyan48@163.com
import numpy as np
import tensorflow as tf
import config

paras = config.get_configs()
fusion_ways = paras['fusion_ways']
fused_nb_feats = paras['fused_nb_feats']
classes = paras['classes']

def sign_sqrt(x):
    # mfb_sign_sqrt = torch.sqrt(F.relu(mfb_out)) - torch.sqrt(F.relu(-mfb_out))
    return tf.keras.backend.sign(x) * tf.keras.backend.sqrt(tf.keras.backend.abs(x) + 1e-10)

def l2_norm(x):
    return tf.keras.backend.l2_normalize(x, axis=-1)


def fusion(x1, x2, way='add'):
    if way == fusion_ways[0]:
        fusion_x = tf.keras.layers.Add()([x1, x2])
    if way == fusion_ways[1]:
        fusion_x = tf.keras.layers.Multiply()([x1, x2])
    if way == fusion_ways[2]:
        fusion_x = tf.keras.layers.Concatenate()([x1, x2])
        fusion_x = tf.keras.layers.Dense(units=fused_nb_feats)(fusion_x)
    if way == fusion_ways[3]:
        fusion_x = tf.keras.layers.Maximum()([x1, x2])
    if way == fusion_ways[4]:
        fusion_x = tf.keras.layers.Average()([x1, x2])
    return fusion_x


def code2net(individual_code, nb_feats=[1024, 2048, 1028]):
    '''
    将一个编码解码为对应的融合网络
    :param pop_code: 一个个体编码
    :param nb_feats:  全部视图的特征个数
    :return: model 融合网络
    '''
    nb_view = (len(individual_code) + 1)//2 #视图个数
    '生成nb_view网络输入'
    input_x = []
    x = []
    x_bn = []
    x_dp = []
    for i in range(nb_view):
        view_id = individual_code[i] #视图编号
        input_x.append(tf.keras.layers.Input((nb_feats[view_id],)))
        x_bn.append(tf.keras.layers.BatchNormalization()(input_x[i]))
        # x_dp.append(tf.keras.layers.Dropout(0.1)(x_bn[i]))
        x.append(tf.keras.layers.Dense(units=fused_nb_feats, activation='relu')(x_bn[i]))

    '融合过程'
    fusion_x = None
    if nb_view == 1:
        fusion_x = x[0]
    else:
        for i in range(nb_view-1):
            way_id = individual_code[nb_view + i]
            if i == 0:
                fusion_x = fusion(x1=x[i], x2=x[i+1], way=fusion_ways[way_id])
            else:
                fusion_x = fusion(x1=fusion_x, x2=x[i+1], way=fusion_ways[way_id])
    fusion_x = tf.keras.layers.BatchNormalization()(fusion_x)
    fusion_x = tf.keras.layers.Lambda(sign_sqrt)(fusion_x)
    fusion_x = tf.keras.layers.Lambda(l2_norm)(fusion_x)
    out_x = tf.keras.layers.Dense(units=classes, activation='softmax')(fusion_x)

    model = tf.keras.models.Model(inputs=input_x, outputs=[out_x])
    return model


if __name__ == '__main__':
    individual_code = [0, 3, 4, 9, 8, 1, 5,
                4, 4, 3, 1, 1, 3]
    model = code2net(individual_code, nb_feats=[1024, 2048, 1028, 512, 256, 512, 256, 1024, 2048, 1028])
    tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)


