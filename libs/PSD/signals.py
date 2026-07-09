import numpy as np
import numpy.typing as npt

def sen(vmax = 1, dc = 0, ff = 1, ph = 0, nn = 100, fs = 1000 ) -> tuple[npt.NDArray[np.number], npt.NDArray[np.number]]:
    '''Generate a sine wave.
        Args:
            vmax: Peak amplitude.
            dc: DC offset.
            ff: Frequency in Hz.
            ph: Phase in radians.
            nn: Number of samples.
            fs: Sampling frequency in Hz.
        Returns:
            xx: Signal samples.
            tt: Time vector.
    '''
    ts = 1/fs
    tt: np.ndarray = np.arange(stop=nn*ts, step=ts)

    xx = vmax * np.sin(2*np.pi*ff*tt + ph) + dc
    xx = np.array(xx).reshape(-1,1)
    tt = tt.reshape(-1,1)

    return xx, tt

def square(vmax = 1, dc = 0, ff = 1, duty = .5, nn = 100, fs = 1000 ) -> tuple[npt.NDArray[np.number], npt.NDArray[np.number]] :
    '''Generates a square wave signal with the given parameters.
        Args:
            vmax: maximum value of the signal. The minimum value will be -vmax. vpp is 2*vmax.
            dc: DC offset of the signal.
            ff: frequency of the signal.
            duty: duty cycle of the signal. It is the percentage of time that the signal is high. It must be between 0 and 1.
            nn: number of samples of the signal.
            fs: sampling frequency of the signal.
        Returns:
            xx: Signal samples.
            tt: Time vector.
    '''
    ts = 1/fs
    tt: np.ndarray = np.arange(stop=nn*ts, step=ts)

    xx = np.where(tt % (1/ff) < (1/ff) * duty, 1, -1)
    xx = vmax * xx + dc
    #signal.square(2* np.pi * ff* tt, duty)

    tt = tt.reshape(-1,1)
    xx = np.array(xx).reshape(-1,1)

    return xx, tt

def sawtooth(vmax = 1, dc = 0, ff = 1, nn = 1, fs = 1000) -> tuple[npt.NDArray[np.number], npt.NDArray[np.number]]:
    '''Generates a sawtooth wave signal with the given parameters.
        Args:
            vmax: maximum value of the signal. The minimum value will be -vmax. vpp is 2*vmax.
        dc: DC offset of the signal.
            ff: frequency of the signal.
            nn: number of samples of the signal.
            fs: sampling frequency of the signal.
        Returns:
            xx: Signal samples.
            tt: Time vector.
    '''
    ts = 1/fs
    tt: np.ndarray = np.arange(stop= nn*ts, step=ts)
    T = 1/ff

    xx = [((vmax/T) * (i%T) + dc) for i in tt]
    #signal.sawtooth(2* np.pi * ff* tt, 0.5)

    tt = tt.reshape(-1,1)
    xx = np.array(xx).reshape(-1,1)

    return xx, tt

def triangle(vmax = 1, dc = 0, ff = 1, duty = 0.5, nn = 1, fs = 1000) -> tuple[npt.NDArray[np.number], npt.NDArray[np.number]]:
    '''Generates a triangle wave signal with the given parameters.
        Args:
            vmax: maximum value of the signal. The minimum value will be -vmax. vpp is 2*vmax.
            dc: DC offset of the signal.
            ff: frequency of the signal.
            duty: duty cycle of the signal. It is the percentage of time that the signal is rising. It must be between 0 and 1.
            nn: number of samples of the signal.
            fs: sampling frequency of the signal.
        Returns:
            xx: Signal samples.
            tt: Time vector.
        '''
    ts = 1/fs
    tt: np.ndarray = np.arange(stop= nn*ts, step=ts)
    T = 1/ff

    xx = np.where(tt % T < duty*T, (vmax/(duty*T)) * (tt % T) + dc, (-vmax/(1-duty)) * (1 - (tt % T)/T) + dc)

    tt = tt.reshape(-1,1)
    xx = np.array(xx).reshape(-1,1)

    return xx, tt

def noise_generator(var = 1, nn = 100, fs = 1000) -> tuple[npt.NDArray[np.number], npt.NDArray[np.number]]:
    '''Generates Gaussian noise, with variance var and zero mean.
        Args:
            var: variance of the noise.
            nn: number of samples of the signal.
            fs: sampling frequency of the signal.
        Returns:
            xx: Noise samples.
            tt: Time vector.
        Note: 
            This distribution corresponds to "white noise". (Extracted from ASYS notes, Stochastic process - White Noise)
    '''
    ts = 1/fs
    tt: np.ndarray = np.arange(stop=nn*ts, step=ts).reshape(-1,1)

    xx = np.random.normal(loc=0, scale=np.sqrt(var), size=nn)
    xx = np.array(xx).reshape(nn,1)

    return xx, tt

def noisy_sen(vmax = 1, dc = 0, ff = 1, ph = 0, nn = 100, fs = 1000, snr = 20) -> tuple[npt.NDArray[np.number], npt.NDArray[np.number]]:
    '''Generates a noisy sine wave signal with the given parameters.
        Args:
            vmax: maximum value of the signal. The minimum value will be -vmax. vpp is 2*vmax.
            dc: DC offset of the signal.
            ff: frequency of the signal.
            ph: phase of the signal in radians.
            nn: number of samples of the signal.
            fs: sampling frequency of the signal.
            snr: signal to noise ratio in dB. It is defined as 10*log10(P_signal/P_noise), where P_signal is the power of the signal and P_noise is the power of the noise. The power of a signal is defined as the mean of the square of the signal.
        Returns:
            xx: Signal samples.
            tt: Time vector.
    '''
    xx, tt = sen(vmax=vmax, dc=dc, ff=ff, ph=ph, nn=nn, fs=fs)
    pot_signal = np.mean(xx**2)
    var_noise = pot_signal / (10**(snr/10))
    
    noise, _ = noise_generator(var_noise, nn, fs)
    xx = xx + noise

    return xx, tt

def kroneker_delta(n : int) -> npt.NDArray[np.number]:
    """
    Función delta de Kronecker.
    """
    delta = np.zeros(n)
    delta[0] = 1

    return delta