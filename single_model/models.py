#encoding=utf-8
#author: liang xinyan
#email: liangxinyan48@163.com
import tensorflow as tf
from single_model import ResnetModel
from config import get_configs
paras = get_configs()
image_size = paras['image_size']
w, h, c = image_size['w'], image_size['h'], image_size['c']

def get_model(model_name='resnet50', classes=1142):
    model = None
    input_x = tf.keras.layers.Input((w, h, c))
    if model_name == 'resnet18':
        model = ResnetModel.ResnetBuilder().build_resnet_18(input_shape=(w, h, c), num_outputs=classes)
    if model_name == 'resnet34':
        model = ResnetModel.ResnetBuilder().build_resnet_34(input_shape=(w, h, c), num_outputs=classes)
    if model_name == 'resnet50':
        model = ResnetModel.ResnetBuilder().build_resnet_50(input_shape=(w, h, c), num_outputs=classes)
    if model_name == 'resnet101':
        model = ResnetModel.ResnetBuilder().build_resnet_101(input_shape=(w, h, c), num_outputs=classes)
    # if model_name == 'resnet50':
    #     model = tf.keras.applications.ResNet50(include_top=True, input_tensor=input_x,
    #                                            classes=classes, weights=None)
    if model_name == 'desnet121':
        model = tf.keras.applications.DenseNet121(include_top=True, input_tensor=input_x,
                                                  classes=classes, weights=None)
    if model_name == 'desnet169':
        model = tf.keras.applications.DenseNet169(include_top=True, input_tensor=input_x,
                                                  classes=classes, weights=None)
    if model_name == 'desnet201':
        model = tf.keras.applications.DenseNet201(include_top=True, input_tensor=input_x,
                                                  classes=classes, weights=None)
    if model_name == 'vgg16':
        model = tf.keras.applications.VGG16(include_top=True, input_tensor=input_x,
                                            classes=classes, weights=None)
    if model_name == 'vgg19':
        model = tf.keras.applications.VGG19(include_top=True, input_tensor=input_x,
                                            classes=classes, weights=None)
    if model_name == 'Xception':
        model = tf.keras.applications.Xception(include_top=True, input_tensor=input_x,
                                               classes=classes, weights=None)
    if model_name == 'InceptionResNetV2':
        model = tf.keras.applications.InceptionResNetV2(include_top=True, input_tensor=input_x,
                                                        classes=classes, weights=None)
    if model_name == 'InceptionV3':
        model = tf.keras.applications.InceptionV3(include_top=True, input_tensor=input_x,
                                                  classes=classes, weights=None)
    if model_name == 'MobileNetV2':
        model = tf.keras.applications.MobileNetV2(include_top=True, input_tensor=input_x,
                                                  classes=classes, weights=None)
    if model_name == 'MobileNet':
        model = tf.keras.applications.MobileNet(include_top=True, input_tensor=input_x,
                                                classes=classes, weights=None)
    if model_name == 'NASNetLarge':
        model = tf.keras.applications.NASNetLarge(include_top=True, input_tensor=input_x,
                                                   classes=classes, weights=None)
    if model_name == 'NASNetMobile':
        model = tf.keras.applications.NASNetMobile(include_top=True, input_tensor=input_x,
                                                    classes=classes, weights=None)
    return model