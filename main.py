import argparse
import json
import matplotlib.pyplot as plt

from train import train


def main(args):
    with open(args.conf, 'r') as f:
        config = json.load(f)

    train(config)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description='Train and validate autonomous car module')

    arg_parser.add_argument(
        '-c',
        '--conf',
        help='path to the configuration file')

    main(arg_parser.parse_args())
