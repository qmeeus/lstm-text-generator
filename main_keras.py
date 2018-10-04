import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

from logger import logger


class Config:
    # INPUT AND OUTPUT FILES

    directory = "data/"
    data = "wonderland.txt"
    encoding = 'utf-8'
    dictionary = "dictionary.pkl"
    features = "features"
    target = "target"
    checkpoint = 'model_checkpoint'

    # NETWORK SETTINGS

    window = 100
    offset = 3
    neurons = 256
    keep_prob = 0.2

    # OPTMIZATION

    loss = 'categorical_crossentropy'
    optimizer = 'adam'
    learning_rate = 0.005
    clip_gradients = 5

    # TRAINING

    n_epochs = 20
    validation_size = 0.1
    batch_size = 128

    # TESTING

    temperatures = (0.0, 0.5, 0.75)
    sample_length = 1000

    def save_path(self, attr):
        return self.directory + getattr(self, attr)


def main():
    config = Config()
    data = load_data(config)
    X, y = preprocessing(data, config)
    model, callbacks = build_model(X, y, config)
    train(model, X, y, callbacks, config)


def load_data(config):
    # load ascii text and covert to lowercase
    with open(config.save_path('data')) as file:
        raw_text = file.read()
        raw_text = raw_text.lower()
    return raw_text


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

    return X, y


def build_model(X, y, config):
    # define the LSTM model
    model = Sequential()
    model.add(LSTM(config.neurons, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(config.keep_prob))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss=config.loss, optimizer=config.optimizer)

    # define the checkpoint
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    return model, callbacks_list


def train(model, X, y, callbacks, config):
    model.fit(X, y, epochs=config.n_epochs, batch_size=config.batch_size, callbacks=callbacks)


def generate(model):
    # load the network weights
    filename = "weights-improvement-19-1.9435.hdf5"
    model.load_weights(filename)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    int_to_char = dict((i, c) for i, c in enumerate(chars))

    # pick a random seed
    start = np.random.randint(0, len(dataX) - 1)
    pattern = dataX[start]
    logger.info("Seed: {}".format("\"", ''.join([int_to_char[value] for value in pattern]), "\""))

    # generate characters
    for i in range(1000):
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_vocab)
        prediction = model.predict(x, verbose=0)
        index = np.argmax(prediction)
        result = int_to_char[index]
        seq_in = [int_to_char[value] for value in pattern]
        sys.stdout.write(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    logger.info("Done.")


if __name__ == '__main__':
    main()
