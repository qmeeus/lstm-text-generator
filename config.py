

class DefaultConfig:

    # INPUT AND OUTPUT FILES
    data_directory = "data/"
    model_directory = "models/"
    data = "wonderland.txt"
    encoding = 'utf-8-sig'
    dictionary = "dictionary.pkl"
    features = "features"
    target = "target"
    checkpoint = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"

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

    sample_length = 1000

    def save_path(self, attr):
        filename = attr if not hasattr(self, attr) else getattr(self, attr)
        return self.model_directory + filename


class Copperfield(DefaultConfig):

    # INPUT AND OUTPUT FILES

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

    sample_length = 1000
