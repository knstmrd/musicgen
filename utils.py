import numpy as np
import librosa as lr
from math import sqrt
import pathlib
import json
from matplotlib import pyplot as plt
from tqdm import tqdm


def create_dirs_write_config(fname, config, predictor_type):
    pathlib.Path('data/output/{}/{}/{}'.format(predictor_type,
                                               config['audio'], fname)).mkdir(parents=True, exist_ok=True)
    with open('data/output/{}/{}/{}/config.json'.format(predictor_type, config['audio'], fname), 'w') as f:
        json.dump(config, f)


def write_keras_model(fname, config, predictor_type, model):
    with open('data/output/{}/{}/{}/keras_config.json'.format(predictor_type, config['audio'], fname), 'w') as f:
        json.dump(model.get_config(), f)


def plot_history(fname, config, predictor_type, history):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.plot(history.history['loss'], 'r-', label='train')
    ax.plot(history.history['val_loss'], 'b-', label='validation')
    ax.legend()

    if predictor_type == 'cnn':
        nhidden = 0
    else:
        nhidden = config['n_hidden']

    fig.savefig('data/output/{}/{}/{}/{}.png'.format(predictor_type,
                                                     config['audio'], fname, nhidden), bbox_inches='tight')


def convert_output_to_audio(output_spectrogram, config, scaler, melfilters, fname, predictor_type):
    output_spectrogram = output_spectrogram[config['use_prev_frames']:, :]  # cut-off seed audio

    print('Min/max output spectrogram {}, {}'.format(np.min(output_spectrogram), np.max(output_spectrogram)))
    output_spectrogram = output_spectrogram.clip(0., 1.)

    output_spectrogram = scaler.inverse_transform(output_spectrogram)
    print('Output spectrogram power range: {} {}'.format(np.min(output_spectrogram), np.max(output_spectrogram)))

    output_spectrogram = invert_mel_db(output_spectrogram.T, melfilters, config).T
    print('Max output amplitude: {}'.format(np.max(output_spectrogram)))

    output = reconstruct_signal_griffin_lim(output_spectrogram, config['framelength'],
                                            config['hop_length'], config['griflim_iter'], config['griflim_stat'])

    if predictor_type == 'cnn':
        nhidden = 0
    else:
        nhidden = config['n_hidden']

    lr.output.write_wav('data/output/{}/{}/{}/p{}_h{}_e{}.wav'.format(predictor_type, config['audio'], fname,
                                                                      config['use_prev_frames'],
                                                                      nhidden, config['n_epochs']),
                        output, config['sr'])


def load_audio(path, sr):
    return lr.load(path, mono=True, sr=sr)


def stft(audio, config):
    return lr.core.stft(audio, n_fft=config['framelength'], hop_length=config['hop_length'])


def load_spectrogram_db(path, config):
    y, sr = load_audio(path, config['sr'])
    return lr.core.amplitude_to_db(stft(y, config))


def load_mel_spectrogram(path, config):
    y, sr = load_audio(path, config['sr'])
    spec = stft(y, config)

    mel_filters = lr.filters.mel(sr, n_fft=config['framelength'],
                                 n_mels=config['n_mel'], fmin=0.0, fmax=None, htk=False, norm=1)
    spec = np.abs(spec)
    print('Max amplitude: {}'.format(np.max(spec)))
    spec = mel_filters.dot(spec ** 2)
    return spec, mel_filters


def invert_mel(spec, mel_filters):
    return mel_filters.T.dot(spec) ** 0.5


def load_mel_spectrogram_db(path, config):
    spec, mel_filters = load_mel_spectrogram(path, config)
    config['ref_power'] = np.max(spec)
    return lr.power_to_db(spec, ref=np.max), mel_filters


def invert_mel_db(spec, mel_filters, config=None):
    if config is None or 'ref_power' not in config.keys():
        ref = 1.0
    else:
        ref = config['ref_power']
    return mel_filters.T.dot(lr.db_to_power(spec, ref=ref)) ** 0.5


def griffinlim(magnitude_spectrogram, fft_size, hopsamp, iterations=100, verbose=False, fast=True):
    """
    As taken from https://github.com/librosa/librosa/issues/434#issuecomment-291266229
    """
    angles = np.exp(2j * np.pi * np.random.rand(*magnitude_spectrogram.shape))

    prev_rebuilt = np.zeros(magnitude_spectrogram.shape)

    t = tqdm(range(iterations), ncols=100, mininterval=2.0, disable=not verbose)
    for i, tt in enumerate(t):
        full = np.abs(magnitude_spectrogram).astype(np.complex) * angles
        inverse = lr.istft(full, hop_length=hopsamp)
        rebuilt = lr.stft(inverse, n_fft=fft_size, hop_length=hopsamp)

        if fast:
            angles = np.exp(1j * np.angle(rebuilt + 0.99 * (rebuilt - prev_rebuilt)))
            prev_rebuilt = rebuilt
        else:
            angles = np.exp(1j * np.angle(rebuilt))

        if verbose:
            diff = np.abs(magnitude_spectrogram) - np.abs(rebuilt)
            t.set_postfix(loss=np.linalg.norm(diff, 'fro'))

    full = np.abs(magnitude_spectrogram).astype(np.complex) * angles
    inverse = lr.istft(full, hop_length=hopsamp)

    return inverse
