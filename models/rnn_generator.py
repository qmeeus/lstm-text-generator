import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from utils.logger import logger

# TODO: Turn to class using tf.estimator API


def build_model(X, y, config):
    # define the LSTM model
    model = Sequential()
    model.add(LSTM(config.neurons, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(Dropout(config.keep_prob))
    model.add(LSTM(config.neurons))
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


def generate(model, dataX, chars, config, length=1000):
    prefix = config.checkpoint[:config.checkpoint.index("{")]
    weights_files = sorted([f for f in os.listdir(config.directory) if f.startswith(prefix)])
    # load the network weights
    filename = config.save_path(weights_files[-1])
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
    for i in range(length):
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

