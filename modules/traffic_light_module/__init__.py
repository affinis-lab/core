import os
import json

from . import model
from . import preprocessing
from . import postprocessing


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load(config):
    default_config = load_default_config()
    default_config.update(config)

    base_path = config['model-dir'] if 'model-dir' in config else BASE_DIR

    path = os.path.join(base_path, default_config['pretrained-model'])

    return {
        'type': 'vector',
        'model': model.build(path),
        'preprocessing': get_preprocessing_fun(
            default_config['image-size']['width'],
            default_config['image-size']['height'],
            default_config['max-predicted-objects']
        ),
        'postprocessing': get_postprocessing_fun(
            default_config['anchors'],
            default_config['number-of-classes'],
            default_config['object-threshold'],
            default_config['nms-threshold']
        ),
    }


def get_preprocessing_fun(image_width, image_height, max_objects):
    return lambda img: preprocessing.fun(img, (image_width, image_height), max_objects)


def get_postprocessing_fun(anchors, num_classes, obj_threshold, nms_threshold):
    return lambda res: postprocessing.fun(res, anchors, num_classes, obj_threshold, nms_threshold)


def load_default_config():
    path = os.path.join(BASE_DIR, 'config.json')
    with open(path, 'r') as f:
        return json.load(f)
