import argparse
import sys

import numpy as np
import pandas
from sklearn import metrics
import tensorflow as tf

import config.text_classification as config
from models.text_classifier import gru_model, lstm_model, bag_of_words_model

# TODO: Turn into robust and scalable model builder
# TODO: Check tf.app.run(main, argv)
# TODO: Check this for migration instructions:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/README.md


def main(unused_argv):
    # TODO: Turn into Trainer / Data / main
    global n_words
    tf.logging.set_verbosity(tf.logging.INFO)

    # Prepare training and testing data
    # TODO: load_dataset deprecated, use tf.data
    # TODO: add support for other datasets (20newsgroup)
    dbpedia = tf.contrib.learn.datasets.load_dataset(
        'dbpedia', test_with_fake_data=FLAGS.test_with_fake_data)
    x_train = pandas.Series(dbpedia.train.data[:, 1])
    y_train = pandas.Series(dbpedia.train.target)
    x_test = pandas.Series(dbpedia.test.data[:, 1])
    y_test = pandas.Series(dbpedia.test.target)

    # Process vocabulary TODO: VocabularyProcessor deprecated, use tf.data
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
        config.MAX_DOCUMENT_LENGTH)

    x_transform_train = vocab_processor.fit_transform(x_train)
    x_transform_test = vocab_processor.transform(x_test)

    x_train = np.array(list(x_transform_train))
    x_test = np.array(list(x_transform_test))

    n_words = len(vocab_processor.vocabulary_)
    print('Total words: {}'.format(n_words))

    # Build model
    # Switch between rnn_model and bag_of_words_model to test different models.
    model_fn = gru_model
    if FLAGS.model == "BOW":
        # Subtract 1 because VocabularyProcessor outputs a word-id matrix where word
        # ids start from 1 and 0 means 'no word'. But
        # categorical_column_with_identity assumes 0-based count and uses -1 for
        # missing word.
        x_train -= 1
        x_test -= 1
        model_fn = bag_of_words_model
    elif FLAGS.model == "LSTM":
        model_fn = lstm_model

    classifier = tf.estimator.Estimator(model_fn=model_fn)

    # Train.
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={config.WORDS_FEATURE: x_train},
        y=y_train,
        batch_size=len(x_train),
        num_epochs=None,
        shuffle=True)
    classifier.train(input_fn=train_input_fn, steps=config.TRAINING_STEPS)

    # Predict.
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={config.WORDS_FEATURE: x_test}, y=y_test, num_epochs=1, shuffle=False)
    predictions = classifier.predict(input_fn=test_input_fn)
    y_predicted = np.array(list(p['class'] for p in predictions))
    y_predicted = y_predicted.reshape(np.array(y_test).shape)

    # Score with sklearn.
    score = metrics.accuracy_score(y_test, y_predicted)
    print('Accuracy (sklearn): {0:f}'.format(score))

    # Score with tensorflow.
    scores = classifier.evaluate(input_fn=test_input_fn)
    print('Accuracy (tensorflow): {0:f}'.format(scores['accuracy']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--test_with_fake_data',
        default=False,
        help='Test the example code with fake data.',
        action='store_true')
    parser.add_argument(
        '--model',
        choices=["GRU", "LSTM", "BOW"],
        help='Which model to use?')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
