import numpy as np
import librosa as lr
from math import sqrt


def load_audio(path, sr):
    return lr.load(path, mono=True, sr=sr)


def stft(audio, config):
    return lr.core.stft(audio, n_fft=config['framelength'], hop_length=config['hop_length'])


def log_abs(spectrogram):
    return np.log(1 + np.absolute(spectrogram))


def load_spectrogram_log(path, config):
    y, sr = load_audio(path, config['sr'])
    return log_abs(stft(y, config))


def load_spectrogram_db(path, config):
    y, sr = load_audio(path, config['sr'])
    return lr.core.power_to_db(stft(y, config))


def load_mel_spectrogram_db(path, config):
    y, sr = load_audio(path, config['sr'])
    return lr.core.power_to_db(lr.feature.melspectrogram(y=y, sr=sr, n_fft=config['framelength'],
                                                         hop_length=config['hop_length']))


def fft_bin_to_hz(n_bin, sample_rate_hz, fft_size):
    """Convert FFT bin index to frequency in Hz.
    Args:
        n_bin (int or float): The FFT bin index.
        sample_rate_hz (int or float): The sample rate in Hz.
        fft_size (int or float): The FFT size.
    Returns:
        The value in Hz.
    """
    n_bin = float(n_bin)
    sample_rate_hz = float(sample_rate_hz)
    fft_size = float(fft_size)
    return n_bin*sample_rate_hz/(2.0*fft_size)


def hz_to_fft_bin(f_hz, sample_rate_hz, fft_size):
    """Convert frequency in Hz to FFT bin index.
    Args:
        f_hz (int or float): The frequency in Hz.
        sample_rate_hz (int or float): The sample rate in Hz.
        fft_size (int or float): The FFT size.
    Returns:
        The FFT bin index as an int.
    """
    f_hz = float(f_hz)
    sample_rate_hz = float(sample_rate_hz)
    fft_size = float(fft_size)
    fft_bin = int(np.round((f_hz*2.0*fft_size/sample_rate_hz)))
    if fft_bin >= fft_size:
        fft_bin = fft_size-1
    return fft_bin


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
        proposal_spectrogram = magnitude_spectrogram * np.exp(1.0j * reconstruction_angle)
        prev_x = x_reconstruct
        x_reconstruct = istft_for_reconstruction(proposal_spectrogram, fft_size, hopsamp)
        prev_x_spec = reconstruction_spectrogram

        diff = sqrt(sum((x_reconstruct - prev_x)**2) / x_reconstruct.size)
        print('Reconstruction iteration: {}/{} RMSE: {} '.format(iterations - n, iterations, diff))
    return x_reconstruct
