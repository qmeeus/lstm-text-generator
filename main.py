import argparse
from utils.logger import logger


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('library', choices=['keras', 'tflearn'])
    args = parser.parse_args()
    return args


def main():
    args = parse_arg()
    if args.library == 'keras':
        logger.info('Using keras')
        from main_keras import main
    elif args.library == 'tflearn':
        logger.info('Using tflearn')
        from main_tflearn import main
    else:
        raise NotImplementedError

    main()


if __name__ == '__main__':
    main()
