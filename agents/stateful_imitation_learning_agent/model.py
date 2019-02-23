from keras.layers.cudnn_recurrent import CuDNNLSTM
from keras.layers import add, concatenate
from keras.layers import Dropout
from keras.layers import Conv2D
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Concatenate
from keras.layers import ReLU
from keras.layers import Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.utils import plot_model
import math

from .utils import eliot_sig, sigmoid


def build(config, plot_core_model=False):
    '''
        Image input
        '''

    return model(config, plot_core_model)


def model(config, plot_core_model):
    height = config['models']['road_seg_module']['image-size']['height']
    width = config['models']['road_seg_module']['image-size']['width']
    channels = config['models']['road_seg_module']['image-channels']

    image_input = Input(shape=(height, width, channels), batch_shape=(8, height, width, channels))

    # Conv block 1
    x = conv_block(image_input, filters=32, kernel=5, stride=4, layer_num=1, pooling=False)
    x = conv_block(x, filters=32, kernel=3, stride=1, layer_num=2, pooling=False)

    # Conv block 2
    x = conv_block(x, filters=64, kernel=3, stride=4, layer_num=3, pooling=False)
    x = conv_block(x, filters=64, kernel=3, stride=1, layer_num=4, pooling=False)

    # Conv block 3
    x = conv_block(x, filters=128, kernel=3, stride=4, layer_num=5, pooling=False)
    x = conv_block(x, filters=128, kernel=3, stride=1, layer_num=6, pooling=False)

    # Conv block 4
    x = conv_block(x, filters=256, kernel=3, stride=1, layer_num=7, pooling=False)
    x = conv_block(x, filters=256, kernel=3, stride=1, layer_num=8, pooling=False)

    x = Flatten()(x)

    # Fully connected layer 1
    x = fc_block(x, units=512, dropuout=0.5, batch_norm=True)

    # One fully connected layer for each input
    num_inputs = 0
    vector_inputs = []
    if config['fc_after']:
        vector_input_fc_layers = []
        for key in config['models'].keys():
            if config['models'][key]['enabled'] and key != 'road_seg_module':
                shape = config['models'][key]['max_obj'] * (4 + 1 + config['models'][key]['num_classes'])

                vector_input = Input(shape=(shape,))

                fc = Dense(units=128, activation='linear', name='input_fc_' + str(key))(vector_input)
                fc = LeakyReLU(alpha=0.01)(fc)
                fc = BatchNormalization(name='input_fc_norm_' + str(key))(fc)
                fc = Dropout(0.3)(fc)

                vector_inputs.append(vector_input)
                vector_input_fc_layers.append(fc)
                num_inputs += 1

        vector_input_fc_layers.append(x)

        # Concatenate segmented image features vector and bounding boxes/confidence/class vectors
        x = Concatenate()(vector_input_fc_layers)

    # All inputs concatenated (with each other and road segmentation module) and forwarded directly to one fully connected layer
    else:
        shape = 0
        for key in config['models'].keys():
            if config['models'][key]['enabled'] and key != 'road_seg_module':
                shape += config['models'][key]['max_obj'] * (4 + 1 + config['models'][key]['num_classes'])
                num_inputs += 1

        vector_input = Input(shape=(shape,), batch_shape=(8,shape))
        vector_inputs.append(vector_input)

        # Concatenate segmented image features vector and bounding boxes/confidence/class vectors
        x = Concatenate()([vector_input, x])

    x = fc_block(x, units=512, dropuout=0.5)
    x = fc_block(x, units=512, dropuout=0.5)

    conv_to_rnn_dims = (8, 64)
    x = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(x)

    x = lstm_block(x)
    x = Flatten()(x)

    steer_output = Dense(units=1, activation=eliot_sig, name='steer_output')(x)

    acc_brake_output = Dense(units=2, activation='softmax', name='acc_brake_output')(x)

    # Output layer
    output = Concatenate()([steer_output, acc_brake_output])

    vector_inputs.append(image_input)
    model = Model(vector_inputs, output)

    if plot_core_model:
        import os
        os.environ["PATH"] += os.pathsep + 'E:\\graphviz-2.38\\release\\bin'

        plot_model(model, to_file='lstm_gru' + str(num_inputs)
                                  + str(config['fc_after']) + '.png',
                   show_shapes=True, show_layer_names=True)

    return model


def conv_block(x, filters, kernel, stride, layer_num, pooling=False):
    x = Conv2D(filters, (kernel, kernel), strides=(stride, stride), padding='same', name='conv_' + str(layer_num), use_bias=False)(x)
    x = BatchNormalization(name='norm_' + str(layer_num))(x)
    x = ReLU()(x)
    if pooling: x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)
    return x


def fc_block(x, units, dropuout, batch_norm=True):
    x = Dense(units=units, activation='linear')(x)
    if batch_norm: x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(dropuout)(x)
    return x


def lstm_block(x):
    # Two layers of bidirectional LSTMs
    # kernel_initializer='he_normal',

    lstm_1 = CuDNNLSTM(512, return_sequences=True,  stateful=False,  name='lstm_1')(x)
    #lstm_1 = Dropout(0.1)(lstm_1)
    lstm_1b = CuDNNLSTM(512, return_sequences=True,  stateful=False, name='lstm_1b')(x)
    #lstm_1b = Dropout(0.1)(lstm_1b)

    x = add([lstm_1, lstm_1b])
    #x = Dropout(0.2)(x)

    lstm_2 = CuDNNLSTM(512, return_sequences=True,  stateful=False, name='lstm_2')(x)
    #lstm_2 = Dropout(0.1)(lstm_2)

    lstm_2b = CuDNNLSTM(512, return_sequences=True,   stateful=False, name='lstm_2b')(x)
    #lstm_2b = Dropout(0.1)(lstm_2b)

    x = concatenate([lstm_2, lstm_2b])

    x = TimeDistributed(Dense(512, activation='relu'), name='output')(x)
    #x = Dropout(0.2)(x)

    return x
