import argparse


def get_args():
    argparser = argparse.ArgumentParser(description='Parse arguments.')

    argparser.add_argument(
        '-c', '--config',
        dest='config',
        metavar='C',
        default='None',
        help='The JSON Configuration file')

    args = argparser.parse_args()

    return args
