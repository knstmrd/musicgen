import numpy as np


def extract_features(spectrogram, config, offset=0):
    # print('Extracting features, ')
    # output = np.zeros((spectrogram.shape[0], config['n_train']))
    output = spectrogram[offset:offset + config['n_train'] * config['step']:config['step'], :]
    return output


def extract_target(spectrogram, config):
    # output = np.zeros((spectrogram.shape[0], config['n_train']))
    output = spectrogram[1:1 + config['n_train'] * config['step']:config['step'], :]
    return output
