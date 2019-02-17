import cv2
import os
from importlib import import_module


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')


def load_agent(name):
    agent = 'agents.' + name
    return import_module(agent)


def load_module(name):
    module = 'modules.' + name
    return import_module(module)


def load_modules(config):
    modules = {}

    for name in config.keys():
        if not config[name]['enabled']:
            continue

        module = load_module(name)
        modules[name] = module.load(config[name])

    return modules


def load_image(dir, name):
    img = cv2.imread(os.path.join(dir, name))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_image(dir, name, img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    filename = os.path.join(dir, name)
    cv2.imwrite(filename, img)
