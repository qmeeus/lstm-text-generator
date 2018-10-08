import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from base.base_model import BaseModel

# TODO: first switch to pure tensorflow then comply with template !!!
class LstmGenerator(BaseModel):

    def __init__(self, data_loader, config):
        super(LstmGenerator, self).__init__(config)
        # Get the data_loader to make the joint of the inputs in the graph
        self.data_loader = data_loader

        # define some important variables
        self.X = None
        self.y = None
        self.is_training = None
        self.out_argmax = None
        self.loss = None
        self.acc = None
        self.optimizer = None
        self.train_step = None

        self.build_model()
        self.init_saver()

    def init_saver(self):
        # just copy the following line in your child class
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

    def build_model(self):
        # Helper Variables
        self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
        self.global_step_inc = self.global_step_tensor.assign(self.global_step_tensor + 1)
        self.global_epoch_tensor = tf.Variable(0, trainable=False, name='global_epoch')
        self.global_epoch_inc = self.global_epoch_tensor.assign(self.global_epoch_tensor + 1)

        # Inputs to the network
        with tf.variable_scope('inputs'):
            self.X, self.y = self.data_loader.get_input()
            self.is_training = tf.placeholder(tf.bool, name='Training_flag')
        tf.add_to_collection('inputs', self.x)
        tf.add_to_collection('inputs', self.y)
        tf.add_to_collection('inputs', self.is_training)

        # Network Architecture
        model = Sequential()
        model.add(LSTM(self.config.neurons, input_shape=(self.X.shape[1], self.X.shape[2]), return_sequences=True))
        model.add(Dropout(self.config.keep_prob))
        model.add(LSTM(self.config.neurons))
        model.add(Dropout(self.config.keep_prob))
        model.add(Dense(self.y.shape[1], activation='softmax'))
        model.compile(loss=self.config.loss, optimizer=self.config.optimizer)

        # define the checkpoint  # TODO: move to init_saver
        filepath = self.config.save_path("checkpoint")
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]

        tf.add_to_collection('train', self.train_step)
        tf.add_to_collection('train', self.loss)
        tf.add_to_collection('train', self.acc)

        return model, callbacks_list