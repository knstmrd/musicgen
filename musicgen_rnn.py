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
                       'bach': 'input/26-Variation_25a2Clav_1955.wav',
                       'contrapunctus': 'input/08_Contrapunctus_VIIIa_3.wav'}

config = {
    'hop_length':  512,
    'framelength': 2048,
    'audio': 'schonberg',
    'n_train': 12000,
    'n_test': 2000,
    'test_offset': 4100,
    'use_prev_frames': 100,
    'start_offset': 0,
    'sr': 22050,
    'batch_size': 128,
    'n_hidden': 80,
    'n_epochs': 40,
    'n_mel': 160
}

spectrogram, melfilters = load_mel_spectrogram_db(original_audio_list[config['audio']], config)
print('Finished loading audio and creating spectrogram, shape: {}\n'.format(spectrogram.shape))
print('Min/max spectrogram: {}, {}'.format(np.min(spectrogram), np.max(spectrogram)))

config['start_offset'] += config['use_prev_frames']
spectrogram = spectrogram.T  # rows should be different training examples, so different points in time
mm_scaler = MinMaxScaler()

spectrogram = mm_scaler.fit_transform(spectrogram)
print('Min/max spectrogram post-scaling: {}, {}'.format(np.min(spectrogram), np.max(spectrogram)))

X_train = np.zeros((config['n_train'], config['use_prev_frames'], spectrogram.shape[1]))

for i in range(config['use_prev_frames']):
    X_train[:, i, :] = spectrogram[i:i + config['n_train'], :]

y_train = spectrogram[config['use_prev_frames']:config['n_train']+config['use_prev_frames'], :]

rnn = keras.models.Sequential()
# rnn.add(keras.layers.SimpleRNN(units=config['n_hidden'],
#                                input_shape=(X_train.shape[1], X_train.shape[2])))
rnn.add(keras.layers.LSTM(units=config['n_hidden'],
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
fig.savefig('output/rnn/{}_{}.png'.format(config['audio'], config['n_hidden']), bbox_inches='tight')

output_spectrogram = np.zeros((config['n_test'] + config['use_prev_frames'], spectrogram.shape[1]))

output_spectrogram[:config['use_prev_frames'], :] = spectrogram[config['test_offset']:config['test_offset']
                                                                                      + config['use_prev_frames'], :]

rnn2 = keras.models.Sequential()
# rnn.add(keras.layers.SimpleRNN(units=config['n_hidden'],
#                                input_shape=(X_train.shape[1], X_train.shape[2])))
rnn2.add(keras.layers.LSTM(units=config['n_hidden'],
                           input_shape=(X_train.shape[1], X_train.shape[2]),
                           batch_input_shape=(1, X_train.shape[1], X_train.shape[2]),
                           stateful=True))

rnn2.add(keras.layers.Dense(config['n_mel']))
rnn2.compile(loss='mse', optimizer='adam')
# rnn2.summary()
rnn2.set_weights(rnn.get_weights())

for i in range(config['n_test']):
    rnn_input = output_spectrogram[i:i + config['use_prev_frames'], :].reshape([1, config['use_prev_frames'], -1])
    rnn_output = rnn2.predict(rnn_input)
    # rnn_output.clip(0., 1.)
    output_spectrogram[config['use_prev_frames'] + i, :] = rnn_output

output_spectrogram = output_spectrogram[config['use_prev_frames']:, :]  # cut-off seed audio

print('Min/max output spectrogram {}, {}'.format(np.min(spectrogram), np.max(spectrogram)))
output_spectrogram.clip(0., 1.)

output_spectrogram = mm_scaler.inverse_transform(output_spectrogram)
print('Output spectrogram power range: {} {}'.format(np.min(output_spectrogram), np.max(output_spectrogram)))

output_spectrogram = invert_mel_db(output_spectrogram.T, melfilters, config).T
print('Max output amplitude: {}'.format(np.max(output_spectrogram)))

output = reconstruct_signal_griffin_lim(output_spectrogram, config['framelength'],
                                        config['hop_length'], 100)

lr.output.write_wav('output/rnn/{}_p{}_h{}_e{}.wav'.format(config['audio'], config['use_prev_frames'],
                                                           config['n_hidden'], config['n_epochs']),
                    output, config['sr'])
