from matplotlib import pyplot as plt
import numpy as np
from utils import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import keras
from matplotlib import pyplot as plt


train_audio_name = 'reich'  # what audio we will train on
gen_audio_name = 'reich'  # what audio will be used as a seed for generation
original_audio_list = {'reich': 'input/01-18_Pulses.wav', 'branca': 'input/04-LightField.wav',
                       'schonberg': 'input/2-05_Phantasy_for_Violin_and_Piano.wav',
                       'bach': 'input/26-Variation_25a2Clav_1955.wav'}

config = {
    'hop_length':  512,
    'framelength': 2048,
    'audio': 'schonberg',
    'db_max': -0.1,
    'n_train': 3000,
    'n_test': 4000,
    'test_offset': 3100,
    'use_prev_frames': 1,
    'gen_verbose': 1000,
    'start_offset': 0,
    'sr': 22050,
    'seed_length': 1,
    'batch_size': 64,
    'n_hidden': 100,
    'n_epochs': 40,
    'n_mel': 160
}

spectrogram, melfilters = load_mel_spectrogram_db(original_audio_list[config['audio']], config)
print('Finished loading audio and creating spectrogram, shape: {}\n'.format(spectrogram.shape))

config['start_offset'] += config['use_prev_frames']
spectrogram = spectrogram.T  # rows should be different training examples, so different points in time
mm_scaler = MinMaxScaler()
spectrogram = mm_scaler.fit_transform(spectrogram)

X_train = spectrogram[:config['n_train'], :]
y_train = spectrogram[1:config['n_train']+1, :]

X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))

rnn = keras.models.Sequential()
rnn.add(keras.layers.SimpleRNN(units=config['n_hidden'],
                               input_shape=(X_train.shape[1], X_train.shape[2])))
rnn.add(keras.layers.Dense(config['n_mel']))
rnn.compile(loss='mse', optimizer='adam')
rnn.summary()

history = rnn.fit(X_train, y_train, epochs=config['n_epochs'], batch_size=config['batch_size'], validation_split=0.1,
                  verbose=1, shuffle=False)

# plot history
fig = plt.figure(figsize=(10, 10))
ax = fig.add_axes([0, 0, 1, 1])
ax.plot(history.history['loss'], 'r-', label='train')
ax.plot(history.history['val_loss'], 'b-', label='validation')
ax.legend()
fig.savefig('out_rnn_{}.png'.format(config['n_hidden']), bbox_inches='tight')

output_spectrogram = np.zeros((config['n_test'] + config['seed_length'], spectrogram.shape[1]))

output_spectrogram[:config['seed_length'], :] = spectrogram[config['test_offset']:config['test_offset']
                                                                                  + config['seed_length'], :]

for i in range(config['n_test']):
    rnn_input = output_spectrogram[config['seed_length'] + i - 1, :].reshape([1, 1, -1])
    rnn_output = rnn.predict(rnn_input)
    rnn_output.clip(0., 1.)
    output_spectrogram[config['seed_length'] + i, :] = rnn_output
#
output_spectrogram.clip(0., 1.)
output_spectrogram = mm_scaler.inverse_transform(output_spectrogram)

output_spectrogram = invert_mel_db(output_spectrogram.T, melfilters).T

output = reconstruct_signal_griffin_lim(output_spectrogram, config['framelength'],
                                        config['hop_length'], 140)

lr.output.write_wav('rnn_{}_{}_{}.wav'.format(config['audio'], config['n_hidden'],
                                              config['n_epochs']), output, config['sr'])
