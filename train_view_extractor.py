#encoding=utf-8
import numpy as np
import argparse
np.random.seed(1337)  # for reproducibility
import tensorflow as tf
import time
from single_model import models
import os
from config import get_configs
from data_utils.data_uitl import get_data
# strategy = tf.distribute.MirroredStrategy()
paras = get_configs()
batch_size = paras['batch_size']
epochs = paras['epochs']
classes = paras['classes']

data_name = paras['data_name']
def main(model_name='resnet'):
    result_save_dir = os.path.join(data_name+'result')
    train_x, train_y, test_x, test_y = get_data(data_base_dir=os.path.join('..', data_name))

    # train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    # test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    # train_ds = train_ds.shuffle(buffer_size=90000).repeat().batch(batch_size).prefetch(buffer_size=90000//batch_size)
    # test_ds = test_ds.shuffle(buffer_size=10000).repeat().batch(batch_size).prefetch(buffer_size=10000 // batch_size)
    # 并行开始
    # with strategy.scope():
    start = time.time()
    warm_up = True
    if warm_up:
        model = models.get_model(model_name, classes=classes)
        # adam = tf.keras.optimizers.SGD(0.8)
        adam = tf.keras.optimizers.Adam()
    else:
        model = tf.keras.models.load_model(os.path.join(data_name+'result', 'best', model_name + '.h5'))
    print(model.summary())
    print(f'max {train_y.shape} {test_y.shape}')
    print(f'Model: {model_name} is training ...')
    print(data_name)
    if warm_up:
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc'])
    # 并行结束
    checkpoint_filepath = os.path.join(result_save_dir, model_name + '.h5')
    # checkpoint_filepath = os.path.join(result_save_dir, model_name + '-{epoch:04d}.h5')
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
    #     checkpoint_filepath, monitor='val_acc', verbose=1, save_weights_only=False)
        checkpoint_filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=10)
    csv_filepath = os.path.join(result_save_dir, model_name + '.csv')
    csv_logger = tf.keras.callbacks.CSVLogger(csv_filepath)
    model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs,
              verbose=1, validation_data=(test_x, test_y),
              callbacks=[csv_logger, early_stop, checkpoint])
    cost_time = time.time()-start
    print('time is:', cost_time)
    with open(model_name+'time.csv', 'a+') as f:
            f.writelines(str(cost_time))

    # model.fit(train_ds, epochs=epochs,
    #           verbose=1, validation_data=test_ds,
    #           callbacks=[csv_logger, early_stop, checkpoint])


if __name__ == "__main__":
    # https: // blog.csdn.net / u010165147 / article / details / 97490500
    # Open-set Recognition: https: // www.wjscheirer.com / projects / openset - recognition /
    # model_name = paras['desnet121']
    # 1. resnet50 2. desnet121 3. MobileNetV2 4. vgg16 5.Xception
    # 6. InceptionV3
    os.makedirs(data_name+'result', exist_ok=True)
    models_ls = ['resnet50', 'desnet121', 'MobileNetV2', 'Xception', 'InceptionV3']
    models_ls += ['resnet18', 'resnet34', 'desnet169', 'desnet201', 'NASNetMobile']
    args = argparse.ArgumentParser()
    args.add_argument('-g', '--gpus', default='0', help='gpus you use here', type=str)
    args.add_argument('-m', '--model', default=0,
                      help='1.resnet50; 2.desnet121; 3.MobileNetV2; 4.vgg16; 5.vgg19;'
                           '6.Xception 7.InceptionV3 8.desnet169', type=int)
    pp = args.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = pp.gpus
    main(model_name=models_ls[pp.model])
#     python train_view_extractor.py -g 2 -m 2
