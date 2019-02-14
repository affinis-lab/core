import argparse
import json

from train import train


def main(args):
    #modules = load_modules()
    #print(modules)

    config_path = args.conf

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    train(config)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description='Train and validate autonomous car module')

    arg_parser.add_argument(
        '-c',
        '--conf',
        help='path to the configuration file')

    main(arg_parser.parse_args())
