import json
import os

from . import model
from . import preprocessing
from . import postprocessing


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load(config):
    default_config = load_default_config()
    print(config)
    default_config.update(config)

    base_path = config['model-dir'] if 'model-dir' in config else BASE_DIR

    path = os.path.join(base_path, default_config['pretrained-model'])

    return {
        'type': 'image',
        'model': model.build(path),
        'preprocessing': get_preprocessing_fun(
            default_config['image-size']['width'],
            default_config['image-size']['height'],
        ),
        'postprocessing': get_postprocessing_fun(),
    }


def get_preprocessing_fun(width, height):
    return lambda img: preprocessing.fun(img, (width, height))


def get_postprocessing_fun():
    return postprocessing.fun


def load_default_config():
    path = os.path.join(BASE_DIR, 'config.json')
    with open(path, 'r') as f:
        return json.load(f)
