import argparse
import json
import matplotlib.pyplot as plt

from train import train
from modules.traffic_light_module.model import *
from modules.traffic_light_module.postprocessing import *


def main(args):
    #modules = load_modules()
    #print(modules)

    config_path = args.conf

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    train(config)


def tl_predict_example(config):
    p = "1.png"
    im = cv2.imread(p)

    # Directly from file
    model = get_model_from_file(config)
    netout = predict_with_model_from_file(config, model, p)

    # From YOLO class
    #model = get_model(config)
    #netout = model.predict(p)

    draw_boxes(im, netout, config['models']['traffic_light_module']['classes'])

    plt.imshow(np.squeeze(im))
    plt.show()


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description='Train and validate autonomous car module')

    arg_parser.add_argument(
        '-c',
        '--conf',
        help='path to the configuration file')

    main(arg_parser.parse_args())
