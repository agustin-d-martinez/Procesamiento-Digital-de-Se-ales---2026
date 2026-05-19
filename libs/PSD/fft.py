import numpy as np
import scipy.signal as signal

from typing import Callable, Literal

WindowType = Literal[
    "rectangular",
    "flattop",
    "hann",
    "hamming",
    "blackman",
    "blackman-harris",
]

WINDOWS: dict[str, Callable[[int], np.ndarray]] = {
    "rectangular": lambda nn: np.ones(nn),
    "flattop": lambda nn: signal.windows.flattop(nn),
    "hann": lambda nn: signal.windows.hann(nn),
    "hamming": lambda nn: np.hamming(nn),
    "blackman": lambda nn: signal.windows.blackman(nn),
    "blackman-harris": lambda nn: signal.windows.blackmanharris(nn)
}

signal.windows.blackmanharris
def fft(xx: np.ndarray, n=None, window: WindowType | None = None) -> np.ndarray:
    '''Returns the fft of the signal xx, normalized by the length of the signal. This is equivalent to np.fft.fft, but it also normalizes the fft by the length of the signal, so that the values are correct.
        Args:
            xx: signal to be transformed.
            n: length of the fft. If n is not given, it is assumed to be the length of the signal.
            window: window function to apply to the signal before taking the fft.
    '''
    nn = xx.shape[0]
    if window is not None:
        try:
            window = WINDOWS[window](nn)
        except KeyError:
            raise ValueError(f'Window {window} not found. Available windows: {list(WINDOWS.keys())}')

        err_window = np.mean(window)
        xx = xx * window/err_window

    return np.fft.fft(xx, n=n, axis=0)/xx.shape[0]

def freq_axis(xx: np.ndarray | int, fs: float) -> np.ndarray:
    '''Equal to np.fft.fftfreq. It generates the correct freq axis for the fft algorithm.
        Axis has the form:
        [0, △f, 2△f, ..., fs/2, -fs/2, -(fs/2-△f), ..., -△f]
        Note: the positive half always starts with 0. The negative half always starts with -fs/2, so, the negative half has one extra value if the length of the signal is even.

        Args:
            xx: Number of points or signal to generate the axis for. Only the length is used.
            fs: sampling frequency
    '''
    if type(xx) == int:
        nn = xx
    else:
        nn = xx.shape[0]
    half = np.arange(0, fs/2, fs/nn)
    ff = np.concatenate((half, -half[::-1] - fs/nn))
    ff = ff[:nn]
    return ff.reshape(-1,1)

def fft_shift(xx: np.ndarray) -> np.ndarray:
    '''Shifts the fft of the signal xx, so that the zero frequency component is in the center of the spectrum. This is equivalent to np.fft.fftshift, but it also normalizes the fft by the length of the signal, so that the values are correct.
    '''
    N = xx.shape[0]
    return np.concatenate((xx[N//2:], xx[:N//2]))
