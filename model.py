import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from tensorflow.contrib import rnn
from utils.logger import logger



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


def generate(model, dataX, chars, config):
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


def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']