from utils import *
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
    'use_prev_frames': 200,
    'start_offset': 0,
    'sr': 22050,
    'batch_size': 128,
    'n_epochs': 20,
    'n_mel': 160,
    'sigmoid_output': False,
    'tensorboard': False,
    'LR_on_plateau': True,
    'freq_filters': True,
    'griflim_iter': 120,
    'griflim_stat': 20,
    'dim': 2
}


def get_model(input_shape):
    cnn = keras.models.Sequential()

    if config['freq_filters']:
        cnn.add(keras.layers.InputLayer(input_shape=(input_shape[0], input_shape[1])))
        cnn.add(keras.layers.Conv1D(filters=512, kernel_size=3, dilation_rate=2))
        cnn.add(keras.layers.Conv1D(filters=512, kernel_size=3, dilation_rate=2))
        cnn.add(keras.layers.Conv1D(filters=config['n_mel'], kernel_size=2, dilation_rate=1))
        cnn.add(keras.layers.GlobalMaxPooling1D())
    else:
        cnn.add(keras.layers.InputLayer(input_shape=(*input_shape, 1)))
        cnn.add(keras.layers.Conv2D(32, (3, 3), dilation_rate=(2, 1), padding='same',
                                    activation='relu'))
        cnn.add(keras.layers.Conv2D(32, (3, 3), dilation_rate=(2, 1), padding='same',
                                    activation='relu'))
        cnn.add(keras.layers.MaxPooling2D(pool_size=(4, 1), strides=(4, 1)))

        cnn.add(keras.layers.Conv2D(32, (3, 3), padding='same',
                                    activation='relu'))
        cnn.add(keras.layers.Conv2D(32, (3, 3), padding='same',
                                    activation='relu'))
        cnn.add(keras.layers.MaxPooling2D(pool_size=(4, 1), strides=(4, 1)))

        cnn.add(keras.layers.Conv2D(64, (3, 3), padding='same',
                                    activation='relu'))
        cnn.add(keras.layers.Conv2D(64, (3, 3), padding='same',
                                    activation='relu'))
        cnn.add(keras.layers.MaxPooling2D(pool_size=(3, 1), strides=(3, 1)))

        cnn.add(keras.layers.Conv2D(64, (3, 3), padding='same',
                                    activation='relu'))
        cnn.add(keras.layers.Conv2D(64, (3, 3), padding='same',
                                    activation='relu'))
        cnn.add(keras.layers.MaxPooling2D(pool_size=(3, 1), strides=(3, 1)))

        cnn.add(keras.layers.Conv2D(1, (1, 1), padding='same',
                                    activation='relu'))
        cnn.add(keras.layers.Flatten())

    opt = keras.optimizers.Adam(lr=0.00125, beta_1=0.9, beta_2=0.999)
    cnn.compile(loss='mse', optimizer=opt)
    cnn.summary()
    return cnn


def get_callbacks(fname):
    callbacks = []

    if config['tensorboard']:
        tbc = keras.callbacks.TensorBoard(log_dir='./data/output/rnn/{}/{}/tb_graphs'.format(config['audio'], fname),
                                          histogram_freq=10,
                                          write_graph=False, write_grads=True,
                                          batch_size=config['batch_size'])
        callbacks.append(tbc)

    if config['LR_on_plateau']:
        rlrp = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, mode='auto',
                                                 min_delta=0.0001, cooldown=1, min_lr=1e-6)
        callbacks.append(rlrp)
    return callbacks


def main():
    fname = str(datetime.now()).replace(':', '_').replace(' ', '_')[5:19]
    create_dirs_write_config(fname, config, 'cnn')

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

    cnn = get_model((X_train.shape[1], X_train.shape[2]))

    write_keras_model(fname, config, 'cnn', cnn)

    callbacks = get_callbacks(fname)
    history = cnn.fit(X_train, y_train, epochs=config['n_epochs'], batch_size=config['batch_size'],
                      validation_split=0.1,
                      verbose=1, callbacks=callbacks)

    plot_history(fname, config, 'cnn', history)

    output_spectrogram = np.zeros((config['n_test'] + config['use_prev_frames'], spectrogram.shape[1]))

    output_spectrogram[:config['use_prev_frames'], :] = spectrogram[config['test_offset']:config['test_offset']
                                                                                          + config['use_prev_frames'], :]

    for i in range(config['n_test']):
        cnn_input = output_spectrogram[i:i + config['use_prev_frames'], :].reshape([1, config['use_prev_frames'],
                                                                                    config['n_mel']])
        cnn_output = cnn.predict(cnn_input)
        cnn_output = cnn_output.clip(0., 1.)
        output_spectrogram[config['use_prev_frames'] + i, :] = cnn_output

    convert_output_to_audio(output_spectrogram, config, mm_scaler, melfilters, fname, 'cnn')


if __name__ == "__main__":
    main()
