import argparse
import cv2
import json
import math
import numpy as np
import os
from datetime import timedelta
from time import time

from utils import load_image
from utils import load_modules
from utils import save_image


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_data(dir, name):
    filename = os.path.join(dir, name)

    with open(filename, 'r') as f:
        data = json.load(f)

    return data


def process(img, model, pre_fun, post_fun):
    x = pre_fun(img)
    res = model.predict(x)[0]
    return post_fun(res)


def main(args):
    with open(args.conf, 'r') as f:
        config = json.load(f)

    modules = load_modules(config['models'])

    input_dir = config['preprocessing']['input-dir']
    input_file = config['preprocessing']['input-file']
    output_dir = config['preprocessing']['output-dir']
    output_file = config['preprocessing']['output-file']

    input_image_dir = os.path.join(input_dir, 'images')
    output_image_dir = os.path.join(output_dir, 'images')

    out = []
    times = []

    data = load_data(input_dir, input_file)
    n = len(data)

    for step, data_point in enumerate(data):
        start = time()
        
        res = {
            'input': [],
            'label': dict(
                (key, value) for key, value in data_point.items() if key != 'image'
            ),
        }

        img = load_image(input_image_dir, data_point['image'])

        for module_name, module in modules.items():
            x = process(
                img=img,
                model=module['model'],
                pre_fun=module['preprocessing'],
                post_fun=module['postprocessing']
            )

            if module['type'] == 'image':
                save_image(output_image_dir, data_point['image'], x)
                x = data_point['image']

            res['input'].append({
                'module_name': module_name,
                'type': module['type'],
                'value': x
            })

        out.append(res)
        times.append(time() - start)

        print_progress(step + 1, n, sum(times) / len(times), n - step - 1)

    filename = os.path.join(output_dir, output_file)
    with open(filename, 'w') as f:
        json.dump(out, f, indent=2)

    print(f'Total time to convert {n} images is {timedelta(seconds=sum(times))}')


def print_progress(step, n, time_per_step, remaining_steps):
    progress = f'Processing image {step}/{n}'
    tps = f'TPS: {time_per_step:.2f}s'
    eta = f'ETA: {timedelta(seconds=remaining_steps * time_per_step)}s'
    print(f'\r{progress}\t\t{tps}\t\t{eta}', end=' ')


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description='Control CARLA vehicle using pretrained agent')

    arg_parser.add_argument(
        '-c',
        '--conf',
        help='path to the configuration file',
        default='config.json',
    )

    main(arg_parser.parse_args())
