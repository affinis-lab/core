import argparse
import json

from utils import load_agent


def train(args):
    with open(args.conf, 'r') as f:
        config = json.load(f)

    agent = load_agent(config['agent']).init(config)
    agent.train()


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description='Train and validate autonomous car module'
    )

    arg_parser.add_argument(
        '-c',
        '--conf',
        help='path to the configuration file',
        default='config.json'
    )

    train(arg_parser.parse_args())
