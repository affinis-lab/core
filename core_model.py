from keras.layers import Dropout, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.utils import plot_model


def get_core_model(config, plot_core_model=False):
    '''
    Image input
    '''

    image_input = Input(shape=(config['models']['road_seg_module']['image_h'], config['models']['road_seg_module']['image_w'], config['models']['road_seg_module']['image_channels']))

    # Conv layer 1
    x = Conv2D(32, (3, 3), strides=(1, 1), padding='same', name='conv_1', use_bias=False)(image_input)
    x = BatchNormalization(name='norm_1')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Conv layer 2
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv_2', use_bias=False)(x)
    x = BatchNormalization(name='norm_2')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Conv layer 3
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv_3', use_bias=False)(x)
    x = BatchNormalization(name='norm_3')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Conv layer 4
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv_4', use_bias=False)(x)
    x = BatchNormalization(name='norm_4')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Conv layer 5
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv_5', use_bias=False)(x)
    x = BatchNormalization(name='norm_5')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)


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

        vector_input = Input(shape=(shape,))
        vector_inputs.append(vector_input)

        # Concatenate segmented image features vector and bounding boxes/confidence/class vectors
        x = Concatenate()([vector_input,x])

    # Fully connected layer 1
    x = Dense(units=128, activation='linear', name='fc_1')(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = BatchNormalization(name='fc_norm_1')(x)
    x = Dropout(0.5)(x)

    # Fully connected layer 2
    x = Dense(units=64, activation='linear', name='fc_2')(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = BatchNormalization(name='fc_norm_2')(x)
    x = Dropout(0.5)(x)

    # Fully connected layer 3
    x = Dense(units=32, activation='linear', name='fc_3')(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = BatchNormalization(name='fc_norm_3')(x)
    x = Dropout(0.5)(x)

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