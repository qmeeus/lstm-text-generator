import numpy as np
from keras.utils import np_utils
from utils.logger import logger


def clean(text):
    to_replace = ["\u2018", "\u2019", "“", "”"]
    for char in to_replace:
        text = text.replace(char, '"')
    return text


def load(config):
    filename = config.data_directory + config.data
    with open(filename, 'r', encoding=config.encoding) as f:
        data = f.read().lower()
    return clean(data)


def preprocessing(data, config):
    # create mapping of unique chars to integers
    chars = sorted(list(set(data)))
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    n_chars = len(data)
    n_vocab = len(chars)
    logger.debug("Total Characters: {}".format(n_chars))
    logger.debug("Total Vocab: {}".format(n_vocab))

    # prepare the dataset of input to output pairs encoded as integers
    window = 100
    dataX = []
    dataY = []
    for i in range(0, n_chars - config.window, 1):
        seq_in = data[i:i + config.window]
        seq_out = data[i + config.window]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])

    n_patterns = len(dataX)
    logger.debug("Total Patterns: {}".format(n_patterns))

    # reshape X to be [samples, time steps, features]
    X = np.reshape(dataX, (n_patterns, config.window, 1))
    # normalize
    X = X / float(n_vocab)
    # one hot encode the output variable
    y = np_utils.to_categorical(dataY)

    return X, y, chars, dataX, dataY