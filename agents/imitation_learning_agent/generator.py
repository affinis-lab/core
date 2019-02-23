import numpy as np
from keras.utils import Sequence
from imgaug import augmenters as iaa

from .utils import load_image


class BatchGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, config, dataset, shuffle=True, jitter = True):
        'Initialization'
        self.config = config
        self.dataset = dataset

        self.image_h = config['models']['road_seg_module']['image-size']['height']
        self.image_w = config['models']['road_seg_module']['image-size']['width']
        self.image_channels = config['models']['road_seg_module']['image-channels']

        self.batch_size = self.config['train']['batch_size']
        self.shuffle = shuffle
        self.jitter = jitter

        self.on_epoch_end()

        st = lambda aug: iaa.Sometimes(0.4, aug)
        oc = lambda aug: iaa.Sometimes(0.3, aug)
        rl = lambda aug: iaa.Sometimes(0.09, aug)
        self.augment = iaa.Sequential([
            rl(iaa.GaussianBlur((0, 1.5))),  # blur images with a sigma between 0 and 1.5
            rl(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05), per_channel=0.5)),  # add gaussian noise to images
            oc(iaa.Dropout((0.0, 0.10), per_channel=0.5)),  # randomly remove up to X% of the pixels
            oc(iaa.CoarseDropout((0.0, 0.10), size_percent=(0.08, 0.2), per_channel=0.5)), # randomly remove up to X% of the pixels

            oc(iaa.Add((-40, 40), per_channel=0.5)),  # change brightness of images (by -X to Y of original value)
            st(iaa.Multiply((0.10, 2.5), per_channel=0.2)),  # change brightness of images (X-Y% of original value)
            rl(iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5)),  # improve or worsen the contrast
            rl(iaa.Grayscale((0.0, 1))),  # put grayscale
        ],
            random_order=True  # do all of the above in random order
        )


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(float(len(self.dataset)) / self.config['train']['batch_size']))


    def __getitem__(self, index):
        'Generate one batch of data'

        image_input = np.zeros((self.batch_size, self.image_h, self.image_w, self.image_channels))
        output = np.zeros((self.batch_size, 3))
        num_inputs = 0

        if not self.config['fc_after']:

            shape = 0
            for key in self.config['models'].keys():
                if self.config['models'][key]['enabled'] and key != 'road_seg_module':
                    shape += self.config['models'][key]['max_obj'] * (4 + 1 + self.config['models'][key]['num_classes'])
                    num_inputs += 1

            vector_input = np.zeros((self.batch_size, shape))

            current_batch = self.dataset[index * self.batch_size:(index + 1) * self.batch_size]
            instance_num = 0
            for instance in current_batch:
                concatenated_vectors = []
                for module in instance['input']:
                    if not self.config['models'][module['module_name']]['enabled']:
                        continue
                    elif module['type'] == 'image':
                        image = load_image(module['value'], self.config)
                        image = self.augment.augment_image(image)
                        image_input[instance_num] = image
                    else:
                        input_num = 0
                        max_obj = self.config['models'][module['module_name']]['max_obj']
                        for i in range(max_obj):
                            if i < len(module['value']):
                                if module['value'][i][5] > 0:
                                    module['value'][i][5] = 1.0
                                elif len(module['value'][i]) > 6:
                                    module['value'][i][6] = 1.0
                                concatenated_vectors += module['value'][i]
                            else:
                                concatenated_vectors += [0 for i in range(4 + 1 + self.config['models'][module['module_name']]['num_classes'])]
                            input_num += 1

                label = [instance['label']['steer'], instance['label']['throttle'], instance['label']['brake']]
                vector_input[instance_num] = concatenated_vectors
                output[instance_num] = label

                instance_num += 1

            return [vector_input, image_input], output

        else:
            vector_inputs = []
            for key in self.config['models'].keys():
                if self.config['models'][key]['enabled'] and key != 'road_seg_module':
                    shape = self.config['models'][key]['max_obj'] * (4 + 1 + self.config['models'][key]['num_classes'])
                    vector_inputs.append(np.zeros((self.batch_size, shape)))
                    num_inputs += 1

            current_batch = self.dataset[index * self.batch_size:(index + 1) * self.batch_size]

            instance_num = 0
            for instance in current_batch:
                detection_module_num = 0
                for module in instance['input']:
                    if not self.config['models'][module['module_name']]['enabled']:
                        continue
                    elif module['type'] == 'image':
                        image = load_image(module['value'], self.config)
                        image = self.augment.augment_image(image)
                        image_input[instance_num] = image
                    else:
                        concatenated_vectors = []
                        input_num = 0
                        max_obj = self.config['models'][module['module_name']]['max_obj']
                        for i in range(max_obj):
                            if i < len(module['value']):
                                if module['value'][i][5] > 0:
                                    module['value'][i][5] = 1.0
                                elif len(module['value'][i]) > 6:
                                    module['value'][i][6] = 1.0
                                concatenated_vectors += module['value'][i]
                            else:
                                concatenated_vectors += [0 for i in range(4 + 1 + self.config['models'][module['module_name']]['num_classes'])]
                            input_num += 1

                        vector_inputs[detection_module_num][instance_num] = concatenated_vectors
                        detection_module_num += 1

                label = [instance['label']['steer'], instance['label']['throttle'], instance['label']['brake']]

                output[instance_num] = label

            return vector_inputs + [image_input], output


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle: np.random.shuffle(self.dataset)


    def normalize(self, image):
        return image/255.


    def size(self):
        return len(self.dataset)