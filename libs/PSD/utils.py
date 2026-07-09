import numpy as np
import numpy.typing as npt

def power(x: npt.NDArray[np.number]) -> float:
    '''power(x: npt.NDArray[np.number]) -> float:
        Returns the power of the signal x, defined as the mean of the square of the Voltage signal. 
    '''
    return np.mean(np.mean(x)**2)

def snr(signal: npt.NDArray[np.number], noise: npt.NDArray[np.number]) -> float:
    return 10 * np.log10(np.mean(signal**2) / np.mean(noise**2))

def autocorrelate(xx: npt.NDArray[np.number]) -> npt.NDArray[np.number]:
    nn = xx.shape[0]

    autocorr = np.correlate(xx, xx, mode='full')/nn         # Normalizated to size. Gives the true values of the autocorrelation
    autocorr = autocorr[autocorr.size//2:]                  # Only keep non negative values
    autocorr = autocorr.reshape(-1,1)                   	# Reshape to column vector (standard of PDS)
    return autocorr

def mod_db(xx: npt.NDArray[np.number]) -> npt.NDArray[np.number]:
    return 20*np.log10(np.abs(xx))

def mod_dbw(xx: npt.NDArray[np.number]) -> npt.NDArray[np.number]:
    return 10*np.log10(np.abs(xx))

def quantizer(xx: npt.NDArray[np.number], Vfs: float, bits: int = 4) -> npt.NDArray[np.number]:
    '''quantizer
    Quantizes the signal xx like an ADC with Vfs and bits.
    xx: Signal to be quantized.
    Vfs: Voltage full scale. ADC: [0, Vfs]
    bits: Number of bits of the ADC.
    ''' 
    q = Vfs/(2**bits) 
    xq = np.round(xx/q) * q

    vmax =  q*(2**(bits-1)-1)
    vmin = -q*2**(bits-1)

    np.clip(xq, vmin, vmax, out=xq)
    return xq

