import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint

from utils.data import load, preprocessing
from utils.logger import logger
from config import Wonderland as Config


def main():
    config = Config()
    data = load(config)
    X, y, chars, dataX, dataY = preprocessing(data, config)
    model, callbacks = build_model(X, y, config)
    train(model, X, y, callbacks, config)
    generate(model, dataX, chars)


def build_model(X, y, config):
    # define the LSTM model
    model = Sequential()
    model.add(LSTM(config.neurons, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(config.keep_prob))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss=config.loss, optimizer=config.optimizer)

    # define the checkpoint
    filepath = config.save_path("checkpoint")
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    return model, callbacks_list


def train(model, X, y, callbacks, config):
    model.fit(X, y, epochs=config.n_epochs, batch_size=config.batch_size, callbacks=callbacks)


def generate(model, dataX, chars):
    # load the network weights
    filename = "data/weights-improvement-20-1.9451.hdf5"
    n_vocab = len(chars)
    model.load_weights(filename)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    int_to_char = dict((i, c) for i, c in enumerate(chars))

    # pick a random seed
    start = np.random.randint(0, len(dataX) - 1)
    pattern = dataX[start]
    logger.info("Seed: {}".format("\"", ''.join([int_to_char[value] for value in pattern]), "\""))

    sample = []

    # generate characters
    for i in range(1000):
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_vocab)
        prediction = model.predict(x, verbose=0)
        index = np.argmax(prediction)
        result = int_to_char[index]
        sample.append(result)
        seq_in = [int_to_char[value] for value in pattern]
        # logger.info(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    logger.info("".join(sample))
    logger.info("Done.")



if __name__ == '__main__':
    main()
