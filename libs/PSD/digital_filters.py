import numpy as np
import numpy.typing as npt

def complementary_delay_filter(x : npt.NDArray[np.number], 
                               x_filtered : npt.NDArray[np.number], 
                               delay: int
                               ) -> npt.NDArray[np.number]:
    """
    Calcula el complemento:
        y = z^(-delay) - H(z)
    x: señal de entrada
    x_filtered: señal filtrada
    delay: retraso del filtro
    """
    hp = np.zeros_like(x_filtered)
    hp[delay:] = x[:-delay] - x_filtered[delay:]

    return hp

def recursive_moving_average(x : npt.NDArray[np.number] , 
                             N : int, 
                             L : int = 1 
                             ) -> npt.NDArray[np.number]:
    """
    Applies a recursive moving average filter to the input signal.

    The filter computes the moving average over a window of ``N`` samples
    using a recursive implementation, reducing the computational cost from
    O(N) to O(1) per sample after initialization.

    Parameters
    ----------
    x : ndarray
        Input signal. It may be one- or multi-dimensional. The filter is
        applied along the first axis.
    N : int
        Length of the moving average window. Must be greater than zero.
    L : int, default=1
        Interpolation factor. A value of ``L=1`` applies no interpolation,
        while larger values insert ``L-1`` samples between consecutive input
        samples before applying the recursive moving average filter.

    Returns
    -------
    ndarray
        Filtered signal with the same shape as the input.
    Notes
    -----
    The recursive implementation follows

        y[n] = y[n-L] + (x[n] - x[n-N*L]) / N

    which is mathematically equivalent to the direct moving average while
    requiring constant computational complexity per output sample.
    """
    y = np.zeros_like(x)

    for n in range(len(x)):
        x_n_N = x[n-N*L] if n >= N*L else 0.0
        y_old = y[n-L] if n >= L else 0.0

        y[n] = y_old + (x[n] - x_n_N)/N             # Funcion de diferencia recursiva para promedio movil
    return y

def delay_signal(x : npt.NDArray[np.number], 
                 delay: int
                 ) -> npt.NDArray[np.number]:
    '''
    Añade un retraso a la señal x. Se implementa como un desplazamiento de la señal hacia adelante, rellenando con ceros al inicio.
    '''
    y = np.zeros_like(x)
    y[delay:] = x[:-delay]
    return y

def lyons_lowpass(x : npt.NDArray[np.number], N : int, C : int = 2, L : int = 1) -> npt.NDArray[np.number]:
    """
    Filtro pasabajos recursivo de Lyons.

    H(z)=(z^(-M)*(1/N)*(1-z^-N)/(1-z^-1))^C

    N debe ser impar
    C debe ser par
    """
    if L == 0:
        raise ValueError("L debe ser mayor que 0.")
    
    # Check imputs
    x = np.asarray(x, dtype=float)

    # Apply the recursive moving average filter C times. 
    tr = x.copy()
    for _ in range(C):
        tr = recursive_moving_average(tr, N, L)

    return tr

def lyons_highpass(x : npt.NDArray[np.number], N : int, C : int = 2, L : int = 1) -> npt.NDArray[np.number]:
    """
    Filtro pasaaltos recursivo de Lyons.

    H(z)=z^(-MC) - (z^(-M)*(1/N)*(1-z^-N)/(1-z^-1))^C

    N debe ser impar
    C debe ser par
    """
    if L == 0:
        raise ValueError("L debe ser mayor que 0.")
    # Check imputs
    x = np.asarray(x, dtype=float)

    # delay value
    delay =  C*L*(N-1)/2
    if delay != int(delay) :
        raise ValueError("El delay M debe ser un entero. Asegúrese de que N sea par y C sea par.")
    delay = int(delay)

    # Apply the recursive moving average filter C times. 
    tr = x.copy()
    for _ in range(C):
        tr = recursive_moving_average(tr, N, L)

    # Delayed input signal
    xd = delay_signal(x, delay)

    return xd - tr

def delay_lyons_filter(N : int, C : int = 2, L : int = 1) -> int:
    """
    Calcula el retraso de un filtro de Lyons.
    """
    if L == 0:
        raise ValueError("L debe ser mayor que 0.")

    delay =  C*L*(N-1)/2
    if delay != int(delay) :
        raise ValueError("El delay M debe ser un entero. Asegúrese de que N sea par y C sea par.")

    return int(delay)