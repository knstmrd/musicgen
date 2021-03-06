from matplotlib import pyplot as plt
import numpy as np
import librosa as lr
from utils import *
import feature_extraction as fe
import scipy
import pandas as pd
from lightgbm.sklearn import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from time import perf_counter
from datetime import datetime
from sklearn.linear_model import LinearRegression
from generation import generate

train_audio_name = 'reich'  # what audio we will train on
gen_audio_name = 'reich'  # what audio will be used as a seed for generation
original_audio_list = {'reich': 'input/01-18_Pulses.wav', 'branca': 'input/04-LightField.wav',
                       'schonberg': 'input/2-05_Phantasy_for_Violin_and_Piano.wav',
                       'bach': 'input/26-Variation_25a2Clav_1955.wav'}

config = {
    'hop_length':  512,
    'framelength': 2048,
    'audio': 'bach',
    'db_max': -0.5,
    'n_train': 4000,
    'step': 2,
    'n_test': 4000,
    'test_offset': 4500,
    'use_prev_frames': 2,
    'gen_verbose': 500,
    'start_offset': 0,
    'sr': 22050,
    'seed_length': 5,
    'n_mel': 120
}

# mls = load_mel_spectrogram_db(original_audio_list[config['audio']], config)
spectrogram, melfilters = load_mel_spectrogram_db(original_audio_list[config['audio']], config)
print('Finished loading audio and creating spectrogram, shape: %d %d\n'%spectrogram.shape)
#

config['start_offset'] += config['use_prev_frames']
spectrogram = spectrogram.T  # rows should be different training examples, so different points in time

X = fe.extract_features(spectrogram, config)
y = fe.extract_target(spectrogram, config)

print('Training data shape: {}, training labels shape: {}'.format(X.shape, y.shape))
start_time = perf_counter()
predictor = LinearRegression(normalize=True)
predictor.fit(X, y)
print(perf_counter() - start_time)
#
output_spectrogram = generate(spectrogram, predictor, config, X.shape[1])
output_spectrogram = invert_mel_db(output_spectrogram.T, melfilters).T

output = reconstruct_signal_griffin_lim(output_spectrogram, config['framelength'],
                                        config['hop_length'], 80)

lr.output.write_wav('test.wav', output, config['sr'])
