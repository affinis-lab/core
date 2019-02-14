import cv2
import os
from importlib import import_module

import numpy as np
import keras
import os
import json
from imgaug import augmenters as iaa


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

def get_available_modules():
    modules_dir = os.path.join(BASE_DIR, 'modules')
    return os.listdir(modules_dir)


def load_module(name):
    module = 'modules.' + name
    return import_module(module)


def load_modules():
    modules = {}

    for name in get_available_modules():
        module = load_module(name)
        modules[name] = {
            'type': module.get_output_type(),
            'model': module.get_model(),
            'preprocessing': module.get_preprocessing_fun(),
            'postprocessing': module.get_postprocessing_fun(),
        }

    return modules


def load_image(path, conf):
    try:
        path = os.path.join(DATA_DIR, 'images', path)

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (conf['models']['road']['image_w'], conf['models']['road']['image_h']))
        return img
    except:
        print("Path  " + path + " not found.")


def save_image(dir, name, img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    filename = os.path.join(dir, 'images', name)
    cv2.imwrite(filename, img)


class BatchGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, config, dataset, shuffle=True, jitter = True):
        'Initialization'
        self.config = config
        self.dataset = dataset

        self.image_h = config['models']['road']['image_h']
        self.image_w = config['models']['road']['image_w']
        self.image_channels = config['models']['road']['image_channels']
        #self.grid_h = config['model']['grid_h']
        #self.grid_w = config['model']['grid_w']

        self.batch_size = self.config['train']['batch_size']
        self.shuffle = shuffle
        self.jitter = jitter

        self.on_epoch_end()

        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        self.aug_pipe = iaa.Sequential(
            [
                # apply the following augmenters to most images
                # iaa.Fliplr(0.5), # horizontally flip 50% of all images
                # iaa.Flipud(0.2), # vertically flip 20% of all images
                # sometimes(iaa.Crop(percent=(0, 0.1))), # crop images by 0-10% of their height/width
                #sometimes(iaa.Affine(
                    # scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                    # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                    # rotate=(-5, 5), # rotate by -45 to +45 degrees
                    # shear=(-5, 5), # shear by -16 to +16 degrees
                    # order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                    # cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                    # mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                #)),
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((0, 3),
                           [
                               # sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                               iaa.OneOf([
                                   iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                                   iaa.AverageBlur(k=(2, 7)),
                                   # blur image using local means with kernel sizes between 2 and 7
                                   iaa.MedianBlur(k=(3, 11)),
                                   # blur image using local medians with kernel sizes between 2 and 7
                               ]),
                               iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
                               # iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                               # search either for all edges or for directed edges
                               # sometimes(iaa.OneOf([
                               #    iaa.EdgeDetect(alpha=(0, 0.7)),
                               #    iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0)),
                               # ])),
                               iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                               # add gaussian noise to images
                               iaa.OneOf([
                                   iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                                   # iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                               ]),
                               # iaa.Invert(0.05, per_channel=True), # invert color channels
                               iaa.Add((-10, 10), per_channel=0.5),
                               # change brightness of images (by -10 to 10 of original value)
                               iaa.Multiply((0.5, 1.5), per_channel=0.5),
                               # change brightness of images (50-150% of original value)
                               iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
                               # iaa.Grayscale(alpha=(0.0, 1.0)),
                               # sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                               # sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))) # sometimes move parts of the image around
                           ],
                           random_order=True
                           )
            ],
            random_order=True
        )


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(float(len(self.dataset)) / self.config['train']['batch_size']))


    def __getitem__(self, index):
        'Generate one batch of data'
        inputs = []
        image_input = np.zeros((self.batch_size, self.image_h, self.image_w, self.image_channels))

        #vector_input = np.zeros((self.batch_size, self.max_objects_per_img * (4 + 1 + self.num_classes)))

        shape = 0
        num_inputs = 0
        for key in self.config['models'].keys():
            if self.config['models'][key]['enabled'] and key != 'road':
                shape += self.config['models'][key]['max_obj'] * (4 + 1 + self.config['models'][key]['num_classes'])
                num_inputs += 1

        vector_input = np.zeros((self.batch_size, shape))

        output = np.zeros((self.batch_size, 3))

        current_batch = self.dataset[index * self.batch_size:(index + 1) * self.batch_size]
        instance_num = 0
        for instance in current_batch:
            concatenated_vectors = []
            for module in instance['input']:
                if module['type'] == 'image':
                    image = load_image(module['value'], self.config)
                    image_input[instance_num] = image
                else:
                    input_num = 0
                    max_obj = self.config['models'][module['module_name']]['max_obj']
                    for i in range(max_obj): #vector in module['value'][:max_obj]:
                        if i < len(module['value']):
                            concatenated_vectors += module['value'][i]
                        else:
                            concatenated_vectors += [0 for i in range(4 + 1 + self.config['models'][module['module_name']]['num_classes'])]
                        input_num += 1

            label = [instance['label']['steer'], instance['label']['throttle'], instance['label']['brake']]
            vector_input[instance_num] = concatenated_vectors
            output[instance_num] = label

            instance_num += 1

        #vector_input = vector_input.reshape((self.batch_size, self.max_objects_per_img * (4 + 1 + self.num_classes)))
        return [vector_input, image_input], output


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle: np.random.shuffle(self.dataset)


    def normalize(self, image):
        return image/255.


    def size(self):
        return len(self.dataset)
