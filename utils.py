import matplotlib.pyplot as plt
import numpy as np


def rms(x):
    """
    Returns root mean square amplitude of x (raises ValueError if NaN).
    """
    out = np.sqrt(np.mean(np.square(x)))
    if np.isnan(out):
        raise ValueError("rms calculation resulted in NaN")
    return out


def get_dbspl(x, mean_subtract=True):
    """
    Returns sound pressure level of x in dB re 20e-6 Pa (dB SPL).
    """
    if mean_subtract:
        x = x - np.mean(x)
    out = 20 * np.log10(rms(x) / 20e-6)
    return out


def set_dbspl(x, dbspl, mean_subtract=True):
    """
    Returns x re-scaled to specified SPL in dB re 20e-6 Pa.
    """
    if mean_subtract:
        x = x - np.mean(x)
    rms_out = 20e-6 * np.power(10, dbspl / 20)
    return rms_out * x / rms(x)


def loguniform(low, high, size=None):
    """
    Draw random samples uniformly on a log scale.
    """
    return np.exp(np.random.uniform(low=np.log(low), high=np.log(high), size=size))


def logspace(start, stop, num):
    """
    Create an array of numbers uniformly spaced on a log scale.
    """
    return np.exp(np.linspace(np.log(start), np.log(stop), num=num))


def freq2erb(freq):
    """
    Convert frequency in Hz to ERB-number.
    Same as `freqtoerb.m` in the AMT.
    """
    return 9.2645 * np.sign(freq) * np.log(1 + np.abs(freq) * 0.00437)


def erb2freq(erb):
    """
    Convert ERB-number to frequency in Hz.
    Same as `erbtofreq.m` in the AMT.
    """
    return (1.0 / 0.00437) * np.sign(erb) * (np.exp(np.abs(erb) / 9.2645) - 1)


def erbspace(start, stop, num):
    """
    Create an array of frequencies in Hz evenly spaced on a ERB-number scale.
    Same as `erbspace.m` in the AMT.

    Args
    ----
    start (float): minimum frequency in Hz
    stop (float): maximum frequency Hz
    num (int): number of frequencies (length of array)

    Returns
    -------
    freqs (np.ndarray): array of ERB-spaced frequencies (lowest to highest) in Hz
    """
    return erb2freq(np.linspace(freq2erb(start), freq2erb(stop), num=num))


def power_spectrum(x, sr, rfft=True, dbspl=True):
    """
    Helper function for computing power spectrum of sound wave.

    Args
    ----
    x (np.ndarray): input waveform (Pa)
    sr (int): sampling rate (Hz)
    rfft (bool): if True, only positive half of power spectrum is returned
    dbspl (bool): if True, power spectrum has units dB re 20e-6 Pa

    Returns
    -------
    fxx (np.ndarray): frequency vector (Hz)
    pxx (np.ndarray): power spectrum (Pa^2 or dB SPL)
    """
    if rfft:
        # Power is doubled since rfft computes only positive half of spectrum
        pxx = 2 * np.square(np.abs(np.fft.rfft(x) / len(x)))
        fxx = np.fft.rfftfreq(len(x), d=1 / sr)
    else:
        pxx = np.square(np.abs(np.fft.fft(x) / len(x)))
        fxx = np.fft.fftfreq(len(x), d=1 / sr)
    if dbspl:
        pxx = 10.0 * np.log10(pxx / np.square(20e-6))
    return fxx, pxx


def complex_tone(
    sr=50e3,
    dur=50e-3,
    freq=220,
    phase=0,
    amplitude=1,
):
    """
    Generate a tone with the specified parameters.
    """
    freq = np.asarray(freq).reshape([-1])
    phase = np.asarray(phase).reshape([-1])
    amplitude = np.asarray(amplitude).reshape([-1])
    assert freq.max() < sr / 2
    assert freq.shape == phase.shape
    assert freq.shape == amplitude.shape
    t = np.arange(0, dur, 1 / sr)
    x = np.zeros_like(t)
    for f, p, a in zip(freq, phase, amplitude):
        x += a * np.sin(2 * np.pi * f * t + p)
    return x


def harmonic_complex_tone(
    sr=50e3,
    dur=50e-3,
    f0=220,
    harmonics=np.arange(1, 11),
    phase="sine",
    amplitudes=None,
):
    """
    Generate a harmonic complex tone with the specified parameters.
    """
    freq = f0 * np.asarray(harmonics).reshape([-1])
    if isinstance(phase, str):
        if phase.lower() == "sine":
            phase = np.zeros_like(freq)
        elif phase.lower() == "cosine":
            phase = (np.pi / 2) * np.ones_like(freq)
        elif phase.lower() == "rand":
            phase = 2 * np.pi * np.random.randn(*freq.shape)
        else:
            raise ValueError(f"unrecognized {phase=}")
    if amplitudes is None:
        amplitude = np.ones_like(freq)
    else:
        amplitude = np.ones_like(freq) * np.asarray(amplitudes)
    x = complex_tone(
        sr=sr,
        dur=dur,
        freq=freq,
        phase=phase,
        amplitude=amplitude,
    )
    return x


def format_axes(
    ax,
    str_title=None,
    str_xlabel=None,
    str_ylabel=None,
    fontsize_title=12,
    fontsize_labels=12,
    fontsize_ticks=12,
    fontweight_title=None,
    fontweight_labels=None,
    xscale="linear",
    yscale="linear",
    xlimits=None,
    ylimits=None,
    xticks=None,
    yticks=None,
    xticks_minor=None,
    yticks_minor=None,
    xticklabels=None,
    yticklabels=None,
    spines_to_hide=[],
    major_tick_params_kwargs_update={},
    minor_tick_params_kwargs_update={},
):
    """
    Helper function for setting axes-related formatting parameters.
    """
    ax.set_title(str_title, fontsize=fontsize_title, fontweight=fontweight_title)
    ax.set_xlabel(str_xlabel, fontsize=fontsize_labels, fontweight=fontweight_labels)
    ax.set_ylabel(str_ylabel, fontsize=fontsize_labels, fontweight=fontweight_labels)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlim(xlimits)
    ax.set_ylim(ylimits)

    if xticks_minor is not None:
        ax.set_xticks(xticks_minor, minor=True)
    if yticks_minor is not None:
        ax.set_yticks(yticks_minor, minor=True)
    if xticks is not None:
        ax.set_xticks(xticks, minor=False)
    if yticks is not None:
        ax.set_yticks(yticks, minor=False)
    if xticklabels is not None:
        ax.set_xticklabels([], minor=True)
        ax.set_xticklabels(xticklabels, minor=False)
    if yticklabels is not None:
        ax.set_yticklabels([], minor=True)
        ax.set_yticklabels(yticklabels, minor=False)

    major_tick_params_kwargs = {
        "axis": "both",
        "which": "major",
        "labelsize": fontsize_ticks,
        "length": fontsize_ticks / 2,
        "direction": "out",
    }
    major_tick_params_kwargs.update(major_tick_params_kwargs_update)
    ax.tick_params(**major_tick_params_kwargs)

    minor_tick_params_kwargs = {
        "axis": "both",
        "which": "minor",
        "labelsize": fontsize_ticks,
        "length": fontsize_ticks / 4,
        "direction": "out",
    }
    minor_tick_params_kwargs.update(minor_tick_params_kwargs_update)
    ax.tick_params(**minor_tick_params_kwargs)

    for spine_key in spines_to_hide:
        ax.spines[spine_key].set_visible(False)

    return ax


def make_power_spectrum_plot(x, sr, figsize=None, **kwargs):
    """ """
    fig, ax = plt.subplots(figsize=figsize)
    if x.ndim == 1:
        x = x[None, ...]
    msg = "expected input shape [timesteps] or [batch, timesteps]"
    assert x.ndim == 2, msg
    for itr in range(x.shape[0]):
        fxx, pxx = power_spectrum(x[itr], sr=sr)
        ax.plot(fxx, pxx)
    kwargs_format_axes = {
        "str_xlabel": "Frequency (Hz)",
        "str_ylabel": "Power (dB)",
        "xscale": "log",
        "yscale": "linear",
        "xlimits": [10, sr / 2],
        "ylimits": [-60, None],
    }
    kwargs_format_axes.update(kwargs)
    ax = format_axes(ax, **kwargs_format_axes)
    return fig, ax


def make_spectrogram_plot(
    x,
    sr,
    nfft=1024,
    noverlap=None,
    vmin=None,
    vmax=None,
    cmap="magma",
    figsize=None,
    str_title=None,
    **kwargs,
):
    """
    Generate figure with a spectrogram
    and a time-aligned sound waveform.
    """
    fig, ax_arr = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        figsize=figsize,
        gridspec_kw={
            "hspace": 0,
            "height_ratios": [1, 5],
        },
        layout="constrained",
    )
    t = np.arange(0, len(x)) / sr
    ax_arr[0].plot(t, x, color="k", lw=1)
    kwargs_format_axes = {
        "str_title": str_title,
        "xticks": [],
        "yticks": [],
        "ylimits": [
            np.min(x) - (np.max(x) - np.min(x)) / 5,
            np.max(x) + (np.max(x) - np.min(x)) / 5,
        ],
        "major_tick_params_kwargs_update": {"length": 0},
    }
    ax_arr[0] = format_axes(ax_arr[0], **kwargs_format_axes)
    ax_arr[0].set_ylabel("Signal", rotation=0, ha="right")
    spectrum, freqs, t, im = ax_arr[1].specgram(
        x,
        NFFT=nfft,
        noverlap=nfft - 1 if noverlap is None else noverlap,
        Fs=sr,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    fig.colorbar(im, label="Intensity (dB)", pad=0.025)
    kwargs_format_axes = {
        "str_xlabel": "Time (s)",
        "str_ylabel": "Frequency (Hz)",
        "xlimits": [t[0], t[-1]],
        "ylimits": [freqs[0], freqs[-1]],
    }
    kwargs_format_axes.update(kwargs)
    ax_arr[1] = format_axes(ax_arr[1], **kwargs_format_axes)
    return fig, ax_arr
