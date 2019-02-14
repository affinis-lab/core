from keras.models import Model, load_model
from keras.layers import Reshape, Lambda, Conv2D, Input, MaxPooling2D, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
import tensorflow as tf
import os
import numpy as np

from modules.traffic_light_module.postprocessing import decode_netout
from modules.traffic_light_module.preprocessing import load_image_predict


class TinyYoloFeature:
    """docstring for ClassName"""
    def __init__(self, input_size):
        input_image = Input(shape=(input_size, input_size, 3))

        # Layer 1
        x = Conv2D(16, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False)(input_image)
        x = BatchNormalization(name='norm_1')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 2 - 5
        for i in range(0,4):
            x = Conv2D(32*(2**i), (3,3), strides=(1,1), padding='same', name='conv_' + str(i+2), use_bias=False)(x)
            x = BatchNormalization(name='norm_' + str(i+2))(x)
            x = LeakyReLU(alpha=0.1)(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 6
        x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_6', use_bias=False)(x)
        x = BatchNormalization(name='norm_6')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

        # Layer 7
        x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_' + str(7), use_bias=False)(x)
        x = BatchNormalization(name='norm_' + str(7))(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 8
        x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_' + str(8), use_bias=False)(x)
        x = BatchNormalization(name='norm_' + str(8))(x)
        x = LeakyReLU(alpha=0.1)(x)

        self.feature_extractor = Model(input_image, x)


class YOLO(object):
    def __init__(self, config):

        self.config = config

        self.image_h = config['models']['traffic_light_module']['image_h']
        self.image_w = config['models']['traffic_light_module']['image_w']

        self.grid_h, self.grid_w = config['models']['traffic_light_module']['grid_h'], config['models']['traffic_light_module']['grid_w']

        self.labels = config['models']['traffic_light_module']['classes']
        self.nb_class = int(len(self.labels))
        self.nb_box = int(len(config['models']['traffic_light_module']['anchors'])/2)
        self.class_wt = np.ones(self.nb_class, dtype='float32')
        self.anchors = config['models']['traffic_light_module']['anchors']

        self.max_box_per_image = config['models']['traffic_light_module']['max_obj']
        #self.batch_size = config['models']['traffic_light_module']['image_h']

        self.obj_thresh = config['models']['traffic_light_module']['obj_thresh']
        self.nms_thresh = config['models']['traffic_light_module']['nms_thresh']

        ##########################
        # Make the model
        ##########################

        # make the feature extractor layers
        input_image = Input(shape=(self.image_h, self.image_w, 3))
        self.true_boxes = Input(shape=(1, 1, 1, self.max_box_per_image, 4))

        self.feature_extractor = TinyYoloFeature(self.image_h).feature_extractor
        features = self.feature_extractor(input_image)

        # Object detection layer
        output = Conv2D(self.nb_box * (4 + 1 + self.nb_class),
                        (1, 1), strides=(1, 1),
                        padding='same',
                        name='DetectionLayer',
                        activation='linear',
                        kernel_initializer='lecun_normal')(features)

        output = Reshape((self.grid_h, self.grid_w, self.nb_box, 4 + 1 + self.nb_class))(output)
        output = Lambda(lambda args: args[0])([output, self.true_boxes])

        self.model = Model([input_image, self.true_boxes], output)
        #self.model.summary()


    def load_weights(self, model_path):
        model = load_model(model_path, custom_objects={'custom_loss': dummy_loss, 'tf': tf})

        idx = 0
        for layer in self.model.layers:
            layer.set_weights(model.get_layer(index=idx).get_weights())
            idx += 1


    def predict(self, image_path):
        image = load_image_predict(image_path, self.image_h, self.image_w)

        dummy_array = np.zeros((1, 1, 1, 1, self.max_box_per_image, 4))
        netout = self.model.predict([image, dummy_array])[0]

        boxes = decode_netout(netout=netout, anchors = self.anchors, nb_class=self.nb_class, obj_threshold=self.obj_thresh, nms_threshold=self.nms_thresh)
        return boxes


    def normalize(self, image):
        return image / 255.


def dummy_loss(y_true, y_pred):
    return tf.sqrt(tf.reduce_sum(y_pred))