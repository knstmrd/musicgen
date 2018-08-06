import numpy as np


def extract_features_single_frame(spectrogram, n_frame, config):
    return spectrogram[n_frame-config['use_prev_frames']:n_frame, :].flatten()


def extract_target(spectrogram, config):
    output = spectrogram[config['start_offset']:config['start_offset'] + config['n_train'] * config['step']:config['step'], :]
    return output


def extract_features(spectrogram, config):
    output = np.zeros((config['n_train'], spectrogram.shape[1] * config['use_prev_frames']))
    for i in range(config['n_train']):
        output[i, :] = extract_features_single_frame(spectrogram, config['start_offset'] + i * config['step'], config)
    return output
