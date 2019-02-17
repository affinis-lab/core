import argparse
import json
import numpy as np

from carla.agent.agent import Agent
from carla.carla_server_pb2 import Control
from carla.driving_benchmark import run_driving_benchmark
from carla.driving_benchmark.experiment_suites import CoRL2017

from utils import load_agent
from utils import load_modules

import numpy as np


class ControlAgent(Agent):

    def __init__(self, model, modules):
        self.model = model
        self.modules = modules

    def process(self, img):
        image_input, vector_inputs = None, []

        for module in self.modules.values():
            x = module['preprocessing'](img)
            res = module['model'].predict(x)[0]
            y = module['postprocessing'](res)

            if module['type'] == 'image':
                image_input = y
            else:
                vector_inputs.append(y)

        return [image_input, np.concatenate(vector_inputs)]

    def predict(self, input_tensor):
        return self.model.predict(input_tensor)[0]

    def run_step(self, measurements, sensor_data, directions, target):
        img = sensor_data['CameraRGB'].data
        res = self.process(img)
        pred = self.predict(res)

        steer, acc, brake = pred[0], pred[1], pred[2]

        control = Control()
        control.steer = steer
        control.throttle = acc
        control.brake = brake

        control.hand_brake = 0
        control.reverse = 0

        return control


def drive(args):
    with open(args.conf.strip(), 'r') as f:
        config = json.load(f)

    modules = load_modules(config['models'])
    model = load_agent('imitation-learning-agent').load('model-02.h5')

    agent = ControlAgent(model, modules)
    experiment_suite = CoRL2017('Town01')
    run_driving_benchmark(agent, experiment_suite)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description='Control CARLA vehicle using pretrained agent')

    arg_parser.add_argument(
        '-c',
        '--conf',
        help='path to the configuration file',
        default='config.json',
    )

    drive(arg_parser.parse_args())
