import cv2
import os
from importlib import import_module


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


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


def load_image(dir, name):
    img = cv2.imread(os.path.join(dir, 'images', name))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_image(dir, name, img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    filename = os.path.join(dir, 'images', name)
    cv2.imwrite(filename, img)
