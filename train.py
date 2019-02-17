import argparse
import json
import matplotlib.pyplot as plt

from utils import load_agent


def train(args):
    with open(args.conf.strip(), 'r') as f:
        config = json.load(f)

    agent = load_agent('imitation-learning-agent').init(config)
    agent.train()


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description='Train and validate autonomous car module')

    arg_parser.add_argument(
        '-c',
        '--conf',
        help='path to the configuration file')

    train(arg_parser.parse_args())
