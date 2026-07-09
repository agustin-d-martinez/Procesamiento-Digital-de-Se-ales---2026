import numpy as np
import scipy.signal as sig

from typing import Callable, Literal
from numpy.typing import ArrayLike

WindowType = Literal[
    "rectangular",
    "flattop",
    "hann",
    "hamming",
    "blackman",
    "blackman-harris",
]

WINDOWS: dict[WindowType, Callable[[int], ArrayLike]] = {
    "rectangular": lambda nn: np.ones(nn),
    "flattop": lambda nn: sig.windows.flattop(nn),
    "hann": lambda nn: sig.windows.hann(nn),
    "hamming": lambda nn: np.hamming(nn),
    "blackman": lambda nn: sig.windows.blackman(nn),
    "blackman-harris": lambda nn: sig.windows.blackmanharris(nn)
}

def fft(xx : ArrayLike,
        nfft : int | None = None,
        window : WindowType | None = None,
        axis : int = -1) -> np.ndarray:
    """
    Compute the normalized Fast Fourier Transform (FFT) of a signal.

    The result is divided by the number of samples so that the spectral
    amplitudes correspond to the original signal amplitudes.

    It also applies a window to the signal before computing the FFT, if specified. The window is amplitude corrected using its mean value.

    Parameters
    ----------
    xx : ArrayLike
        Input signal.
    n : int, optional
        Number of FFT points. If omitted, the signal length is used.
    window : WindowType, optional
        Window applied before computing the FFT. The window is amplitude
        corrected using its mean value.

    Returns
    -------
    ndarray
        Normalized FFT of the input signal.
    """
    xx = np.asarray(xx)
    N = xx.shape[-1]
    if window is not None:
        try:
            window = WINDOWS[window](N)
        except KeyError:
            raise ValueError(f'Window {window} not found. Available windows: {list(WINDOWS.keys())}')

        err_window = np.mean(window)
        xx = xx * window/err_window

    return np.fft.fft(xx, n=nfft, axis=axis)/N

def freq_axis(xx: ArrayLike | int, fs: float) -> np.ndarray:
    """
    Generate the frequency axis associated with an FFT.

    The returned axis follows the ordering used by ``numpy.fft.fft``:
        [0, Δf, ..., fs/2, -fs/2, ..., -Δf]

    Parameters
    ----------
    xx : ArrayLike or int
        Input signal or number of samples.
    fs : float
        Sampling frequency.

    Returns
    -------
    ndarray
        Frequency vector with the same length as the FFT output.
    """    
    if isinstance(xx, int):
        N = xx
    else:
        xx = np.asarray(xx)
        N = xx.shape[-1]
    half = np.arange(0, fs/2, fs/N)
    ff = np.concatenate((half, -half[::-1] - fs/N))
    ff = ff[:N]
    return ff

def fft_shift(xx: ArrayLike) -> np.ndarray:
    """
    Shift the zero-frequency component to the center of the spectrum.

    This function is equivalent to ``numpy.fft.fftshift``.

    Parameters
    ----------
    xx : ArrayLike
        Spectrum or FFT result.

    Returns
    -------
    ndarray
        Shifted spectrum.
    """
    N = xx.shape[-1]
    return np.concatenate((xx[N//2:], xx[:N//2]))

def blackman_tukey(xx : ArrayLike, 
                   fs : float = 1, 
                   M : int = None, 
                   nfft : int = None
                   ) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate the Power Spectral Density (PSD) using the Blackman-Tukey method.

    The algorithm computes the autocorrelation of the input signal, truncates
    it to the desired lag length, applies a Blackman window, and computes the
    Fourier transform of the windowed autocorrelation.

    Parameters
    ----------
    xx : ArrayLike
        Input signal.
    fs : float, default=1
        Sampling frequency.
    M : int, optional
        Maximum correlation lag. If omitted, ``N // 5`` is used.
    nfft : int, optional
        Number of FFT points. If omitted, the signal length is used.

    Returns
    -------
    f : ndarray
        Frequency vector.
    Pxx : ndarray
        Estimated power spectral density.
    """
    xx = np.asarray(xx)
    N = np.max(xx.shape)     # Cant muestras

    if nfft is None:
        nfft = N
    if M is None:
        M = N//5

    xx = xx - np.mean(xx)

    # 1. correlation    
    r = autocorrelate(xx, mode='full')
    r = r[N-M-1 : N+M]
    # Note: correlate returns a 2N-1 vector. Blackman-Tuckey wants the 2M+1 (the center part).
    #       The function trim the ends of lenght N-M-1.

    # 2. window
    win = sig.windows.blackman(len(r))
    
    # 3. FFT
    Pxx = np.abs(np.fft.rfft(r*win, n=nfft))
    Pxx[1:-1] *= 2                              # Correction positive frequency
    Pxx /= fs                                   # Normalization to fs

    f_ = np.fft.rfftfreq(nfft, 1/fs)
    return f_, Pxx

def autocorrelate(xx: ArrayLike, 
                  mode : Literal['full', 'positive'] = 'positive',
                  axis : int = -1) -> np.ndarray:
    """
    Compute the autocorrelation of a signal.

    The autocorrelation is normalized by the signal length.

    Parameters
    ----------
    xx : ArrayLike
        Input signal(s). If two-dimensional, each row is treated as an
        independent signal.
    mode : {"full", "positive"}, default="positive"
        Portion of the autocorrelation sequence to return.

        - ``"full"`` returns the complete sequence, including negative
          and positive lags.
        - ``"positive"`` returns only the non-negative lags, starting
          from zero.
    axis : int, default=-1
        Axis along which the autocorrelation is computed.

    Returns
    -------
    ndarray
        Autocorrelation sequence.
    """
    xx = np.asarray(xx)

    N = xx.shape[axis]
    xx = np.moveaxis(xx, axis, -1)

    corr_lenght = 2 * N - 1 if mode == "full" else N
    corr = np.empty(xx.shape[:-1] + (corr_lenght,))

    for index in np.ndindex(xx.shape[:-1]):
        r = sig.correlate(xx[index], xx[index], mode="full") / N

        if mode == "positive":
            r = r[N-1:]
        elif mode != "full":
            raise ValueError("mode must be either 'full' or 'positive'.")

        corr[index] = r

    corr = np.moveaxis(corr, -1, axis)

    return corr