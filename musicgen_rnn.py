from utils import get_callbacks, create_dirs_write_config, load_mel_spectrogram_db, write_keras_model
from utils import plot_history, convert_output_to_audio
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import keras
from datetime import datetime


original_audio_list = {'reich': 'input/01-18_Pulses.wav', 'branca': 'input/04-LightField.wav',
                       'schonberg': 'input/2-05_Phantasy_for_Violin_and_Piano.wav',
                       'bach': 'input/26-Variation_25a2Clav_1955.wav',
                       'contrapunctus': 'input/08_Contrapunctus_VIIIa_3.wav',
                       'allegro': 'input/1-03_Water_Music_Suite_No_1_inFMajHWV_348_III_AllegroAndanteAllegro.wav'}

config = {
    'hop_length':  256,
    'framelength': 1024,
    'audio': 'contrapunctus',
    'n_train': 20480,
    'n_test': 3000,
    'test_offset': 4100,
    'use_prev_frames': 95,
    'start_offset': 0,
    'sr': 22050,
    'batch_size': 128,
    'n_hidden': 200,
    'n_layers': 4,
    'n_epochs': 150,
    'n_mel': 160,
    'rnn_type': 'gru',
    'sigmoid_output': False,
    'tensorboard': False,
    'save_full_model': True,
    'LR_on_plateau': True,
    'earlystopping': True,
    'griflim_iter': 120,
    'griflim_verbose': True,
    'griflim_fast': True,
}


def get_model(input_shape, batch_input_shape=None):
    rnn = keras.models.Sequential()

    if batch_input_shape is not None:
        rnn.add(keras.layers.InputLayer(batch_input_shape=batch_input_shape))
        for i in range(config['n_layers']-1):
            if config['rnn_type'] == 'lstm':
                rnn.add(keras.layers.LSTM(units=config['n_hidden'],
                                          batch_input_shape=batch_input_shape,
                                          stateful=True, return_sequences=True))
            elif config['rnn_type'] == 'gru':
                rnn.add(keras.layers.GRU(units=config['n_hidden'],
                                         batch_input_shape=batch_input_shape,
                                         stateful=True, return_sequences=True))

        if config['rnn_type'] == 'lstm':
            rnn.add(keras.layers.LSTM(units=config['n_hidden'],
                                      batch_input_shape=batch_input_shape,
                                      stateful=True))
        elif config['rnn_type'] == 'gru':
            rnn.add(keras.layers.GRU(units=config['n_hidden'],
                                     batch_input_shape=batch_input_shape,
                                     stateful=True))
    else:
        rnn.add(keras.layers.InputLayer(input_shape=input_shape))
        if config['rnn_type'] == 'lstm':
            rnn.add(keras.layers.LSTM(units=config['n_hidden']))
        elif config['rnn_type'] == 'gru':
            rnn.add(keras.layers.GRU(units=config['n_hidden']))

    rnn.add(keras.layers.Dense(config['n_mel']))

    if config['sigmoid_output']:
        rnn.add(keras.layers.Activation('sigmoid'))

    # opt = keras.optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999)
    opt = keras.optimizers.RMSprop(lr=0.0015)
    rnn.compile(loss='mse', optimizer=opt)
    rnn.summary()
    return rnn


def main():
    fname = str(datetime.now()).replace(':', '_').replace(' ', '_')[5:19]
    create_dirs_write_config(fname, config, 'rnn')

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

    write_keras_model(fname, config, 'rnn', rnn)

    callbacks = get_callbacks(fname, config)
    history = rnn.fit(X_train, y_train, epochs=config['n_epochs'], batch_size=config['batch_size'],
                      validation_split=0.1,
                      verbose=1, shuffle=False, callbacks=callbacks)

    plot_history(fname, config, 'rnn', history)

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

    convert_output_to_audio(output_spectrogram, config, mm_scaler, melfilters, fname, 'rnn')


if __name__ == "__main__":
    main()
