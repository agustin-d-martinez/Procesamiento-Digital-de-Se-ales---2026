"""
PDS - Digital Signal Processing library.

Collection of tools for signal generation, spectral analysis,
digital filtering and visualization.
"""

from .signals import (
    sin,
    square,
    sawtooth,
    triangle,
    noise_generator,
    noisy_sin,
    kronecker_delta,
    delay_signal,
)

from .spectral import (
    fft,
    fft_shift,
    freq_axis,
    autocorrelate,
    blackman_tukey,
)

from .filters import (
    complementary_delay_filter,
    recursive_moving_average,
    lyons_lowpass,
    lyons_highpass,
    delay_lyons_filter,
)

from .plot import (
    plot_pzmap,
    plot_filt_mag,
    plot_filt_phase,
    plot_filt_delay,
    plot_filt_resp,
    plot_template,
)

from .utils import (
    power,
    snr,
    voltage_db,
    power_db,
    quantizer,
)


__all__ = [
    # Signals
    "sin",
    "square",
    "sawtooth",
    "triangle",
    "noise_generator",
    "noisy_sin",
    "kronecker_delta",
    "delay_signal",

    # Spectral
    "fft",
    "fft_shift",
    "freq_axis",
    "autocorrelate",
    "blackman_tukey",

    # Filters
    "complementary_delay_filter",
    "recursive_moving_average",
    "lyons_lowpass",
    "lyons_highpass",
    "delay_lyons_filter",

    # plot
    "plot_pzmap",
    "plot_filt_mag",
    "plot_filt_phase",
    "plot_filt_delay",
    "plot_filt_resp",
    "plot_template",

    # utils
    "power",
    "snr",
    "voltage_db",
    "power_db",
    "quantizer",
]