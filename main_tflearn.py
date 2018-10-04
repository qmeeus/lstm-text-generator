import os
import pickle
import numpy as np
import tflearn

from logger import logger


class Config:

    # INPUT AND OUTPUT FILES

    directory = "data/"
    data = "copperfield.txt"
    encoding = 'utf-8-sig'
    dictionary = "dictionary.pkl"
    features = "features"
    target = "target"
    checkpoint = 'model_checkpoint'

    # NETWORK SETTINGS

    window = 30
    offset = 3
    neurons = 512
    keep_prob = 0.5

    # OPTMIZATION

    loss = 'categorical_crossentropy'
    optimizer = 'adam'
    learning_rate = 0.005
    clip_gradients = 5

    # TRAINING

    n_epochs = 50
    validation_size = 0.1
    batch_size = 64

    # TESTING

    temperatures = (0.0, 0.5, 0.75)
    sample_length = 1000

    def save_path(self, attr):
        return self.directory + getattr(self, attr)


def main():
    config = Config()
    data = load_data(config)
    logger.debug('text size: {}'.format(len(data)))
    dictionary, X, y = preprocessing(data, config)
    model = build_model(dictionary, config)
    train(model, X, y, data, config)


def load_data(config):
    with open(config.save_path('data'), 'r', encoding=config.encoding) as f:
        data = f.read()
    return data


def preprocessing(data, config):
    dict_file = config.save_path('dictionary')
    if os.path.exists(dict_file):
        with open(dict_file, "rb") as file:
            dictionary = pickle.load(file)
    else:
        chars = sorted(set(data))
        dictionary = {c: i for i, c in enumerate(chars)}
        with open(dict_file, "wb") as file:
            pickle.dump(dictionary, file)

    m, n = int(np.ceil((len(data) - config.window) / config.offset)), len(dictionary)

    logger.debug("m={},n={}".format(m, n))

    features = []
    target = []

    for i in range(0, len(data) - config.window, config.offset):
        features.append(data[i:i + config.window])
        target.append(data[i + config.window])

    X = np.zeros((m, config.window, n), dtype=np.bool_)
    y = np.zeros((m, n), dtype=np.bool_)

    for i, (xi, yi) in enumerate(zip(features, target)):
        for j in range(config.window):
            X[i, j, dictionary[xi[j]]] = 1
        y[i, dictionary[yi]] = 1

    np.save(config.save_path('features'), X)
    np.save(config.save_path('target'), y)

    return dictionary, X, y


def build_model(dictionary, config):
    dict_size = len(dictionary)

    net = tflearn.input_data(shape=(None, config.window, dict_size))

    net = tflearn.lstm(net, config.neurons, return_seq=True)
    net = tflearn.dropout(net, config.keep_prob)

    net = tflearn.lstm(net, config.neurons, return_seq=True)
    net = tflearn.dropout(net, config.keep_prob)

    net = tflearn.lstm(net, config.neurons, return_seq=False)
    net = tflearn.dropout(net, config.keep_prob)

    net = tflearn.fully_connected(net, dict_size, activation='softmax')

    net = tflearn.regression(
        net,
        optimizer=config.optimizer,
        loss=config.loss,
        learning_rate=config.learning_rate
    )

    model = tflearn.SequenceGenerator(
        net,
        dictionary=dictionary,
        seq_maxlen=config.window,
        clip_gradients=config.clip_gradients,
        checkpoint_path=config.save_path('checkpoint')
    )

    return model


def train(model, X, y, data, config):
    for epoch in range(config.n_epochs):
        model.fit(X, y, validation_set=config.validation_size, batch_size=config.batch_size, n_epoch=1)
        logger.info("Epoch {}".format(epoch))
        test(model, data, config)


def test(model, data, config):
    for temperature in config.temperatures:
        i = np.random.randint(0, len(data) - config.window)
        seed = data[i:i + config.window]
        sample_text = model.generate(seq_length=config.sample_length, temperature=temperature, seq_seed=seed)
        logger.info("Temperature: {}".format(temperature))
        logger.info(sample_text)


if __name__ == '__main__':
    main()
