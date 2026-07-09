import numpy as np
import matplotlib.pyplot as plt
import numpy.typing as npt

def plot_pzmap(z: list[complex], p: list[complex], ax : plt.Axes = None, *args, **kwargs) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(*args, **kwargs)
    plt.scatter(np.real(z), np.imag(z), color='b', marker='o')
    plt.scatter(np.real(p), np.imag(p), color='r', marker='x')
    circle = plt.Circle((0,0), 1, fill=False)
    ax.add_patch(circle)
    ax.set_xlim((-1.1, 1.1))
    ax.set_ylim((-1.1, 1.1))

    ax.axhline(0, color='black')
    ax.axvline(0, color='black')
    plt.grid(True, which='both', ls='--')
    plt.title('Polos y Ceros del Filtro')
    plt.ylabel('Real')
    plt.xlabel('Imaginario')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    return ax

def plot_filt_mag(f : npt.NDArray[np.number], H : npt.NDArray[np.number], 
                  *, 
                  ax : plt.Axes | None = None, 
                  title : str | None = None, 
                  label : str | None = None,
                  **kwargs,
                  ) -> plt.Line2D:
    if ax is None:
        ax = plt.gca()

    mag = 20*np.log10(np.maximum(np.abs(H), 1e-13))
    line, = ax.plot(f, mag, label=label, **kwargs)
    ax.set_title(title)
    ax.set_ylabel('Magnitud [dB]')
    ax.set_xlabel('Frecuencia [Hz]')
    ax.grid(True, which='both', ls='--')

    if label is not None:
        ax.legend()

    return line

def plot_filt_phase(f : npt.NDArray[np.number], H : npt.NDArray[np.number], 
                    *,
                    ax : plt.Axes | None = None, 
                    title : str | None = None, 
                    label : str | None = None,
                    **kargs,
                    ) -> plt.Line2D:
    if ax is None:
        ax = plt.gca()

    phase = np.unwrap(np.angle(H))
    line, =ax.plot(f, phase, label=label, **kargs)
    ax.set_title(title)
    ax.set_ylabel('Fase [rad]')
    ax.set_xlabel('Frecuencia [Hz]')
    ax.grid(True, which='both', ls='--')

    if label is not None:
        ax.legend()

    return line

def plot_filt_delay(f : npt.NDArray[np.number], H : npt.NDArray[np.number], 
                    *,
                    ax : plt.Axes | None = None, 
                    title : str | None = None, 
                    label : str | None = None,
                    **kargs,
                    ) -> plt.Line2D:
    if ax is None:
        ax = plt.gca()

    phase = np.unwrap(np.angle(H))
    gd = -np.diff(phase)/(2*np.pi*np.diff(f))
    gd = np.append(gd, gd[-1])

    line, = ax.plot(f, gd, label=label, **kargs)
    ax.set_title(title)
    ax.set_ylabel('Retardo [s]')
    ax.set_xlabel('Frecuencia [Hz]')
    ax.grid(True, which='both', ls='--')

    if label is not None:
        ax.legend()

    return line

def plot_filt_resp(f : npt.NDArray[np.number], H : npt.NDArray[np.number],
                   *,
                   ax : plt.Axes | None = None, 
                   title : str | None = None, 
                   label : str | None = None, 
                   **kargs
                   ) -> None:
    if ax is None:
        fig, ax = plt.subplots(3, 1, sharex=True)

    plot_filt_mag(f, H, title, ax[0], label, **kargs)
    plot_filt_phase(f, H, title, ax[1], label, **kargs)
    plot_filt_delay(f, H, title, ax[2], label, **kargs)


def plot_template(  fpass : float | list[float], 
                    fstop : float | list[float], 
                    attpass : float | list[float] = 0.5,
                    attstop : float | list[float] = 40) -> None: 
    """
    Plotea una plantilla de diseño de filtro digital.

    Parameters
    -----------
    fpass : float o tupla
        Frecuencia de paso o tupla de frecuencias de paso para los filtros 'bandpass' o 'bandstop'.
    ripple : float
        Máxima ondulación en la banda de paso (en dB). Por defecto es 0.5 dB.
    fstop : float o tupla
        Frecuencia de detención o tupla de frecuencias de detención para los filtros 'bandpass' o 'bandstop'.
    attenuation : float
        Atenuación mínima en la banda de detención (en dB). Por defecto es 40 dB.
        
    Returns
    --------
    None
        
    Example
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from PSD.plot import plot_template
    >>> fig_id, axes_hdl = bodePlot(H1, fig_id=1, axes_hdl='none', filter_description='Filtro pasa bajos', worN=1000, digital=False, xaxis='omega', fs=2*np.pi)
    >>> plt.sca(axes_hdl[0])
    >>> fpass = [1, 20, 30]
    >>> fstop = [10, 15, 40]
    >>> attepass = [10, 1]
    >>> attestop = [20, 20]
    >>> PSD.plot.plot_template(fpass, fstop, attepass, attestop)
    >>> plt.legend()

    >>> fpass = 10
    >>> fstop = 50
    >>> plt.xlim(0, 100)
    >>> plt.ylim(-50, 10)
    >>> PSD.plot.plot_template(fpass, fstop)
    >>> plt.legend()

    """

    # Axis limits
    xmin, xmax, ymin, ymax = plt.axis()

    puntos = []
    if isinstance(fpass, (int, float)):
        puntos.append((fpass, 'pass'))
    else:
        puntos.extend((f, 'pass') for f in fpass)
    if isinstance(fstop, (int, float)):
        puntos.append((fstop, 'stop'))
    else:
        puntos.extend((f, 'stop') for f in fstop)
    puntos.sort(key=lambda x: x[0])

    # Range of pass bands and stop bands
    ranges = []
    if puntos[0][0] > xmin:
        ranges.append((xmin, puntos[0][0], puntos[0][1]))
    for a, b in zip(puntos[:-1], puntos[1:]):
        if a[1] == b[1]:
            ranges.append((a[0], b[0], a[1]))
    if puntos[-1][0] < xmax:
        ranges.append((puntos[-1][0], xmax, puntos[-1][1]))

    if isinstance(attpass, (int, float)):
        attpass = [attpass] * len(ranges)
    if isinstance(attstop, (int, float)):
        attstop = [attstop] * len(ranges)

    # Range fill 
    i_pass, i_stop = 0, 0
    for f1, f2, tipo in ranges:
        if tipo == 'pass':
            plt.fill([f1, f1, f2, f2], [ymin, -attpass[i_pass], -attpass[i_pass], ymin], 'lightgrey', alpha=0.4, hatch='x', lw=1, ls='--', ec='k', 
                     label='Plantilla' if i_pass == 0 else None)
            i_pass += 1
        else:
            plt.fill([f1, f2, f2, f1], [-attstop[i_stop], -attstop[i_stop], ymax, ymax], 'lightgrey', alpha=0.4, hatch='x', lw=1, ls='--', ec='k')
            i_stop += 1
