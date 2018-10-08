# LSTM Text Generator

Inspired from [this article](https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/)

## Objective
The objective of this project is to explore techniques for automatic text generation using LSTM recurrent neural networks.
The current implementation uses [keras](https://keras.io/) but it might move to a pure tensorflow implementation at some point and follow the best practices and structures laid out in [this template](https://github.com/MrGemy95/Tensorflow-Project-Template).

A [docker](https://docker.com/) container is provided for the project portability. I might explore possibilities of deployment using [Kubeflow](https://www.kubeflow.org/) as well.

## Requirements
- docker
- nvidia-drivers and nvidia-docker for GPU support 

## Data
The model is built on two freely available books from the [gutenberg project](http://www.gutenberg.org/):
- [Alice's Adventures in Wonderland by Lewis Carroll](http://www.gutenberg.org/ebooks/11).
- [David Copperfield by Charles Dickens](http://www.gutenberg.org/ebooks/766).

To add your own data, just move the text file to `data` and specify the name of the file with the extension when training the model.
If necessary, add your own configuration in `config.py`.

## Installation
To build the container with CPU support, simply run the following command from a system with docker installed:

```docker build -t lstm --build-arg user_id=$(id -u) -f Dockerfile .```

Then running the container:

```docker run -d --name lstm -v $(pwd):/home/patrick/src/ lstm wonderland.txt```

To build the container with GPU support, use the following command from a system with nvidia-docker and nvidia-drivers installed:

```nvidia-docker build -t lstm --build-arg user_id=$(id -u) -f Dockerfile.gpu .```

Next, run the container with:

```nvidia-docker run -d --name lstm -v $(pwd):/home/patrick/src/ lstm wonderland.txt```

**NB: on Linux systems, make sure that you are part of docker group. If this is not the case, run `sudo usermod -aG docker $USER`**
