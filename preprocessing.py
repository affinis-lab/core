import cv2
import json
import numpy as np
import os

from utils import load_modules
from utils import save_image


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUT_DIR = os.path.join(BASE_DIR, 'out')


# TODO: docstrings; command line arguments


def load_data(name):
    filename = os.path.join(DATA_DIR, name)

    with open(filename, 'r') as f:
        data = json.load(f)

    return data


def load_image(dir, name):
    img = cv2.imread(os.path.join(dir, 'images', name))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def process(model, pre_fun, post_fun, data, image=False):
    filename = data['image']
    img = load_image(DATA_DIR, filename)
    x = pre_fun(img)
    x = np.expand_dims(x, 0)

    res = model.predict(x)[0]
    res = post_fun(res)

    if image:
        save_image(OUT_DIR, filename, res)
        return filename

    return res


def main():
    modules = load_modules()

    out = []
    for data_point in load_data('train.json')[73:74]:
        res = {
            'input': [],
            'label': dict(
                (key, value) for key, value in data_point.items() if key != 'image'
            ),
        }

        for module_name, module in modules.items():
            res['input'].append({
                'module_name': module_name,
                'type': module['type'],
                'value': process(
                    model=module['model'],
                    pre_fun=module['preprocessing'],
                    post_fun=module['postprocessing'],
                    data=data_point,
                    image=module['type'] == 'image'
                )
            })

        out.append(res)

    filename = os.path.join(OUT_DIR, 'trainval.json')
    with open(filename, 'w') as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
