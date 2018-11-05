import numpy as np
import librosa as lr
from math import sqrt
import pathlib
import json
from matplotlib import pyplot as plt


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
    fig.savefig('data/output/{}/{}/{}/{}.png'.format(predictor_type,
                                                     config['audio'], fname, config['n_hidden']), bbox_inches='tight')


def convert_output_to_audio(output_spectrogram, config, scaler, melfilters, fname, predictor_type):
    output_spectrogram = output_spectrogram[config['use_prev_frames']:, :]  # cut-off seed audio

    print('Min/max output spectrogram {}, {}'.format(np.min(output_spectrogram), np.max(output_spectrogram)))
    output_spectrogram = output_spectrogram.clip(0., 1.)

    output_spectrogram = scaler.inverse_transform(output_spectrogram)
    print('Output spectrogram power range: {} {}'.format(np.min(output_spectrogram), np.max(output_spectrogram)))

    output_spectrogram = invert_mel_db(output_spectrogram.T, melfilters, config).T
    print('Max output amplitude: {}'.format(np.max(output_spectrogram)))

    output = reconstruct_signal_griffin_lim(output_spectrogram, config['framelength'],
                                            config['hop_length'], 80)

    lr.output.write_wav('data/output/{}/{}/{}/p{}_h{}_e{}.wav'.format(predictor_type, config['audio'], fname,
                                                                      config['use_prev_frames'],
                                                                      config['n_hidden'], config['n_epochs']),
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


def stft_for_reconstruction(x, fft_size, hopsamp):
    """Compute and return the STFT of the supplied time domain signal x.
    Args:
        x (1-dim Numpy array): A time domain signal.
        fft_size (int): FFT size. Should be a power of 2, otherwise DFT will be used.
        hopsamp (int):
    Returns:
        The STFT. The rows are the time slices and columns are the frequency bins.
    """
    window = np.hanning(fft_size)
    fft_size = int(fft_size)
    hopsamp = int(hopsamp)
    return np.array([np.fft.rfft(window*x[i:i+fft_size])
                     for i in range(0, len(x)-fft_size, hopsamp)])


def istft_for_reconstruction(X, fft_size, hopsamp):
    """Invert a STFT into a time domain signal.
    Args:
        X (2-dim Numpy array): Input spectrogram. The rows are the time slices and columns are the frequency bins.
        fft_size (int):
        hopsamp (int): The hop size, in samples.
    Returns:
        The inverse STFT.
    """
    fft_size = int(fft_size)
    hopsamp = int(hopsamp)
    window = np.hanning(fft_size)
    time_slices = X.shape[0]
    len_samples = int(time_slices*hopsamp + fft_size)
    x = np.zeros(len_samples)
    for n, i in enumerate(range(0, len(x)-fft_size, hopsamp)):
        x[i:i+fft_size] += window*np.real(np.fft.irfft(X[n]))
    return x


def reconstruct_signal_griffin_lim(magnitude_spectrogram, fft_size, hopsamp, iterations):
    """Reconstruct an audio signal from a magnitude spectrogram.
    Given a magnitude spectrogram as input, reconstruct
    the audio signal and return it using the Griffin-Lim algorithm from the paper:
    "Signal estimation from modified short-time fourier transform" by Griffin and Lim,
    in IEEE transactions on Acoustics, Speech, and Signal Processing. Vol ASSP-32, No. 2, April 1984.
    Args:
        magnitude_spectrogram (2-dim Numpy array): The magnitude spectrogram. The rows correspond to the time slices
            and the columns correspond to frequency bins.
        fft_size (int): The FFT size, which should be a power of 2.
        hopsamp (int): The hope size in samples.
        iterations (int): Number of iterations for the Griffin-Lim algorithm. Typically a few hundred
            is sufficient.
    Returns:
        The reconstructed time domain signal as a 1-dim Numpy array.
    """
    time_slices = magnitude_spectrogram.shape[0]
    len_samples = int(time_slices*hopsamp + fft_size)
    x_reconstruct = np.random.randn(len_samples)
    n = iterations  # number of iterations of Griffin-Lim algorithm.
    while n > 0:
        n -= 1
        reconstruction_spectrogram = stft_for_reconstruction(x_reconstruct, fft_size, hopsamp)
        if n == iterations - 1:
            prev_x_spec = reconstruction_spectrogram

        reconstruction_angle = np.angle(reconstruction_spectrogram + 0.99 * (reconstruction_spectrogram - prev_x_spec))
        # reconstruction_angle = np.angle(reconstruction_spectrogram)
        proposal_spectrogram = magnitude_spectrogram * np.exp(1.0j * reconstruction_angle)
        prev_x = x_reconstruct
        x_reconstruct = istft_for_reconstruction(proposal_spectrogram, fft_size, hopsamp)
        prev_x_spec = reconstruction_spectrogram

        diff = sqrt(sum((x_reconstruct - prev_x)**2) / x_reconstruct.size)
        print('Reconstruction iteration: {}/{} RMSE: {} '.format(iterations - n, iterations, diff))
    return x_reconstruct
