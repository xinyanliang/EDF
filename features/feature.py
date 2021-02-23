#encoding=utf-8
#author: liang xinyan
#email: liangxinyan48@163.com
import tensorflow as tf
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '7'
def sign_sqrt(x):
    # mfb_sign_sqrt = torch.sqrt(F.relu(mfb_out)) - torch.sqrt(F.relu(-mfb_out))
    return tf.keras.backend.sign(x) * tf.keras.backend.sqrt(tf.keras.backend.abs(x) + 1e-10)

def l2_norm(x):
    return tf.keras.backend.l2_normalize(x, axis=-1)


class Feature:
    def __init__(self, model_dir='models', save_data_dir='database', database_name='database'):
        self.model_dir = model_dir
        self.save_data_dir = save_data_dir
        self.database_name = database_name

    def get_feats_single_view(self, view_model, x, save_data_suffix='train_X'):
        model_h5_fn = os.path.join(self.model_dir, view_model + '.h5')
        model = tf.keras.models.load_model(model_h5_fn)
        model_feat = tf.keras.models.Model(inputs=model.input, outputs=model.layers[-2].output)
        x_feats = model_feat.predict(x=x)
        np.save(os.path.join(self.save_data_dir, view_model + save_data_suffix + '.npy'), x_feats)
        return x_feats

    def get_feats_multi_views(self, view_models, x, save_data_suffix='train_X'):
        if save_data_suffix is None:
            save_data_suffix = ''
        views = []
        for view_model in view_models:
            x_feats = self.get_feats_single_view(view_model, x=x, save_data_suffix=save_data_suffix)
            views.append(x_feats)
        return views

    def get_input_edf(self, code, views):
        code_list = [int(i) for i in code.strip().split('-')]
        view_code = code_list[:len(code_list) // 2 + 1]
        edf_input_views = []
        for i in code_list:
            edf_input_views.append(views[i])
        return edf_input_views

    def get_feats_by_edf(self, views, save_data_suffix='train_X', edf_model_name='3-2-0-1-0-4-0'):
        if save_data_suffix is None:
            save_data_suffix = ''
        model_h5_fn = os.path.join(self.model_dir, edf_model_name + '.h5')
        print(model_h5_fn)
        _custom_objects = {"sign_sqrt": sign_sqrt, 'l2_norm': l2_norm, 'tf': tf}
        model = tf.keras.models.load_model(model_h5_fn, custom_objects=_custom_objects)
        model_feat = tf.keras.models.Model(inputs=model.input, outputs=model.layers[-2].output)
        edf_input_views = self.get_input_edf(code=edf_model_name, views=views)
        x_feats = model_feat.predict(x=edf_input_views)
        np.save(os.path.join(self.save_data_dir, edf_model_name + save_data_suffix + '.npy'), x_feats)
        return x_feats

