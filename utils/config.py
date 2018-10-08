from configparser import ConfigParser


def read_config(config_file):
    config = ConfigParser()
    config.read(config_file)
    return config
