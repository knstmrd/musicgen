import numpy as np
from utils import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import keras
from matplotlib import pyplot as plt
import json
from datetime import datetime
import pathlib


original_audio_list = {'reich': 'input/01-18_Pulses.wav', 'branca': 'input/04-LightField.wav',
                       'schonberg': 'input/2-05_Phantasy_for_Violin_and_Piano.wav',
                       'bach': 'input/26-Variation_25a2Clav_1955.wav',
                       'contrapunctus': 'input/08_Contrapunctus_VIIIa_3.wav',
                       'allegro': 'input/1-03_Water_Music_Suite_No_1_inFMajHWV_348_III_AllegroAndanteAllegro.wav'}

config = {
    'hop_length':  256,
    'framelength': 1024,
    'audio': 'bach',
    'n_train': 20480,
    'n_test': 8000,
    'test_offset': 4100,
    'use_prev_frames': 20,
    'start_offset': 0,
    'sr': 22050,
    'batch_size': 128,
    'n_hidden': 256,
    'n_epochs': 1,
    'n_mel': 100,
    'tensorboard': False,
    'LR_on_plateau': True
}


def get_model(input_shape, batch_input_shape=None):
    rnn = keras.models.Sequential()

    if batch_input_shape is not None:
        rnn.add(keras.layers.LSTM(units=config['n_hidden'],
                                  input_shape=input_shape,
                                  batch_input_shape=batch_input_shape,
                                  stateful=True, return_sequences=True))
        rnn.add(keras.layers.LSTM(units=config['n_hidden'],
                                  input_shape=input_shape,
                                  batch_input_shape=batch_input_shape,
                                  stateful=True))
    else:
        rnn.add(keras.layers.LSTM(units=config['n_hidden'],
                                  input_shape=input_shape))

    rnn.add(keras.layers.Dense(config['n_mel']))
    # rnn.add(keras.layers.Activation('sigmoid'))
    adam = keras.optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999)
    rnn.compile(loss='mse', optimizer=adam)
    rnn.summary()
    return rnn


def create_dirs(fname):
    pathlib.Path('data/output/rnn/{}/{}'.format(config['audio'], fname)).mkdir(parents=True, exist_ok=True)


def get_callbacks(fname):
    callbacks = []

    if config['tensorboard']:
        tbc = keras.callbacks.TensorBoard(log_dir='./data/output/rnn/{}/{}/tb_graphs'.format(config['audio'], fname),
                                          histogram_freq=10,
                                          write_graph=False, write_grads=True,
                                          batch_size=config['batch_size'])
        callbacks.append(tbc)

    if config['LR_on_plateau']:
        rlrp = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=0, mode='auto',
                                                 min_delta=0.0001, cooldown=1, min_lr=0)
        callbacks.append(rlrp)
    return callbacks


def main():
    fname = str(datetime.now()).replace(':', '_').replace(' ', '_')[5:19]
    create_dirs(fname)
    with open('data/output/rnn/{}/{}/config.json'.format(config['audio'], fname), 'w') as f:
        json.dump(config, f)

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

    rnn = get_model((X_train.shape[1], X_train.shape[2]),
                    batch_input_shape=(config['batch_size'], X_train.shape[1], X_train.shape[2]))

    callbacks = get_callbacks(fname)
    history = rnn.fit(X_train, y_train, epochs=config['n_epochs'], batch_size=config['batch_size'],
                      validation_split=0.1,
                      verbose=1, shuffle=False, callbacks=callbacks)

    # plot history
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.plot(history.history['loss'], 'r-', label='train')
    ax.plot(history.history['val_loss'], 'b-', label='validation')
    ax.legend()
    fig.savefig('data/output/rnn/{}/{}/{}.png'.format(config['audio'], fname, config['n_hidden']), bbox_inches='tight')

    output_spectrogram = np.zeros((config['n_test'] + config['use_prev_frames'], spectrogram.shape[1]))

    output_spectrogram[:config['use_prev_frames'], :] = spectrogram[config['test_offset']:config['test_offset']
                                                                                          + config['use_prev_frames'], :]

    rnn2 = get_model((X_train.shape[1], X_train.shape[2]),
                     batch_input_shape=(1, X_train.shape[1], X_train.shape[2]))

    rnn2.set_weights(rnn.get_weights())

    for i in range(config['n_test']):
        rnn_input = output_spectrogram[i:i + config['use_prev_frames'], :].reshape([1, config['use_prev_frames'], -1])
        rnn_output = rnn2.predict(rnn_input)
        rnn_output = rnn_output.clip(0., 1.)
        output_spectrogram[config['use_prev_frames'] + i, :] = rnn_output

    output_spectrogram = output_spectrogram[config['use_prev_frames']:, :]  # cut-off seed audio

    print('Min/max output spectrogram {}, {}'.format(np.min(output_spectrogram), np.max(output_spectrogram)))
    output_spectrogram = output_spectrogram.clip(0., 1.)

    output_spectrogram = mm_scaler.inverse_transform(output_spectrogram)
    print('Output spectrogram power range: {} {}'.format(np.min(output_spectrogram), np.max(output_spectrogram)))

    output_spectrogram = invert_mel_db(output_spectrogram.T, melfilters, config).T
    print('Max output amplitude: {}'.format(np.max(output_spectrogram)))

    output = reconstruct_signal_griffin_lim(output_spectrogram, config['framelength'],
                                            config['hop_length'], 80)

    lr.output.write_wav('data/output/rnn/{}/{}/p{}_h{}_e{}.wav'.format(config['audio'], fname,
                                                                       config['use_prev_frames'],
                                                                       config['n_hidden'], config['n_epochs']),
                        output, config['sr'])


if __name__ == "__main__":
    main()
