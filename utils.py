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
