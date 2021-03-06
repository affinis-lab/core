import argparse
import json
import numpy as np
import cv2

from carla.agent.agent import Agent
from carla.carla_server_pb2 import Control
from carla.driving_benchmark import run_driving_benchmark
from carla.driving_benchmark.experiment_suites import CoRL2017
from functools import reduce

from keras.models import load_model

from utils import load_agent
from utils import load_modules

import numpy as np
import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
JUNK_DIR = os.path.join(BASE_DIR, 'junk')


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
                vector_inputs += y

        vector_inputs = [reduce(lambda x,y: x + y, vector_inputs)]

        return [vector_inputs, image_input]

    def predict(self, input_tensor):
        return self.model.predict(input_tensor)[0]

    def run_step(self, measurements, sensor_data, directions, target):
        img = sensor_data['CameraRGB'].data
        res = self.process(img)
        res[1] = np.expand_dims(res[1], 0)

        pred = self.predict(res)

        steer, acc, brake = pred[0], pred[1], pred[2]

        control = Control()
        control.steer = steer

        if acc < 0.3:
            control.throttle = 0
        elif acc < 0.7:
            control.throttle = 0.5
        else:
            control.throttle = 1

        speed = measurements.player_measurements.forward_speed
        if speed * 3.6 >= 20:
            control.throttle = 0
        
        if brake < 0.3:
            control.brake = 0
        elif brake < 0.7:
            control.brake = 0.5
        else:
            control.brake = 1
        
        control.hand_brake = 0
        control.reverse = 0

        return control


def drive(args):
    with open(args.conf.strip(), 'r') as f:
        config = json.load(f)

    modules = load_modules(config['models'])
    model = load_agent('imitation_learning_agent').load('lstm-2.h5')

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
