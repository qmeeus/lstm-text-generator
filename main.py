import argparse
from utils import data
from models import rnn_generator


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('book', type=str)
    parser.add_argument('--config', type=str)
    args = parser.parse_args()
    return args


args = parse_arg()
if args.book == 'wonderland':
    from config import DefaultConfig as Config
elif args.book == 'copperfield':
    from config import Copperfield as Config
else:
    raise NotImplementedError

config = Config()
data = data.load(config)
X, y, chars, dataX, dataY = data.preprocessing(data, config)
model, callbacks = rnn_generator.build_model(X, y, config)
rnn_generator.train(model, X, y, callbacks, config)
rnn_generator.generate(model, dataX, chars, config)

