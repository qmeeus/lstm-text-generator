import argparse
from utils.data import load, preprocessing
from model import build_model, train, generate


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('book', choices=['wonderland', 'copperfield'])
    args = parser.parse_args()
    return args


def main():
    args = parse_arg()
    if args.book == 'wonderland':
        from config import Wonderland as Config
    elif args.book == 'copperfield':
        from config import Copperfield as Config
    else:
        raise NotImplementedError

    config = Config()
    data = load(config)
    X, y, chars, dataX, dataY = preprocessing(data, config)
    model, callbacks = build_model(X, y, config)
    train(model, X, y, callbacks, config)
    generate(model, dataX, chars, config)


if __name__ == '__main__':
    main()
