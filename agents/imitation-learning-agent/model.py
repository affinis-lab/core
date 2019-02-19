from keras.layers import Dropout
from keras.layers import Conv2D
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Lambda
from keras.layers import Concatenate
from keras.layers import ReLU
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.utils import plot_model


def build(config, plot_core_model=False):
    '''
    Image input
    '''

    height = config['models']['road_seg_module']['image-size']['height']
    width = config['models']['road_seg_module']['image-size']['width']
    channels = config['models']['road_seg_module']['image-channels']
    image_input = Input(shape=(height, width, channels))


    # Conv block 1
    x = conv_block(image_input, filters=32, kernel=5, stride=2, layer_num=1, pooling=False)
    x = conv_block(x, filters=32, kernel=3, stride=1, layer_num=2)

    # Conv block 2
    x = conv_block(x, filters=64, kernel=3, stride=2, layer_num=3, pooling=False)
    x = conv_block(x, filters=64, kernel=3, stride=1, layer_num=4)

    # Conv block 3
    x = conv_block(x, filters=128, kernel=3, stride=2, layer_num=5, pooling=False)
    x = conv_block(x, filters=128, kernel=3, stride=1, layer_num=6)

    # Conv block 4
    x = conv_block(x, filters=256, kernel=3, stride=1, layer_num=7, pooling=False)
    x = conv_block(x, filters=256, kernel=3, stride=1, layer_num=8, pooling=False)

    x = Flatten()(x)

    # Fully connected layer 1
    x = fc_block(x, units=512, dropuout=0.5, batch_norm=True)


    '''
    Concatenate road segmentation image wither other modules inputs
    '''

    # One fully connected layer for each input
    num_inputs=0
    vector_inputs = []
    if config['fc_after']:
        vector_input_fc_layers = []
        for key in config['models'].keys():
            if config['models'][key]['enabled'] and key != 'road_seg_module':
                shape = config['models'][key]['max_obj'] * (4 + 1 + config['models'][key]['num_classes'])

                vector_input = Input(shape=(shape,))

                fc = fc_block(vector_input, units=128, dropuout=0.5, batch_norm=True)
                fc = fc_block(fc, units=128, dropuout=0.5, batch_norm=True)

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

        vector_input = Input(shape=(shape,))
        vector_inputs.append(vector_input)

        # Concatenate segmented image features vector and bounding boxes/confidence/class vectors
        x = Concatenate()([vector_input,x])


    # Fully connected layer 2
    x = fc_block(x, units=512, dropuout=0.5, batch_norm=True)
    #x = fc_block(x, units=512, dropuout=0.5, batch_norm=True)

    # Output layer
    output = Dense(units=3, activation='softmax', name='output')(x)

    vector_inputs.append(image_input)

    model = Model(vector_inputs, output)

    if plot_core_model:
        import os
        os.environ["PATH"] += os.pathsep + 'E:\\graphviz-2.38\\release\\bin'

        plot_model(model, to_file='model' + str(num_inputs)
                                  + str(config['fc_after']) + '.png',
                   show_shapes=True, show_layer_names=True)

    return model


def conv_block(x, filters, kernel, stride, layer_num, pooling = True):
    x = Conv2D(filters, (kernel, kernel), strides=(stride, stride), padding='same', name='conv_' + str(layer_num), use_bias=False)(x)
    x = BatchNormalization(name='norm_' + str(layer_num))(x)
    x = LeakyReLU(alpha=0.1)(x)
    #x = ReLU()(x)
    if pooling: x = MaxPooling2D(pool_size=(2, 2))(x)
    return x


def fc_block(x, units, dropuout, batch_norm=True):
    x = Dense(units=units, activation='linear')(x)
    x = LeakyReLU(alpha=0.01)(x)
    #x = ReLU()(x)
    if batch_norm: x = BatchNormalization()(x)
    x = Dropout(dropuout)(x)
    return x
