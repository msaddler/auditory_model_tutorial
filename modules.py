import numpy as np
import torch

import filters
import utils


class FIRFilterbank(torch.nn.Module):
    def __init__(self, fir, dtype=torch.float32, **kwargs_conv1d):
        """
        Finite impulse response (FIR) filterbank

        Args
        ----
        fir (list or np.ndarray or torch.Tensor):
            Filter coefficients. Shape (n_taps,) or (n_filters, n_taps)
        dtype (torch.dtype):
            Data type to cast `fir` to in case it is not a `torch.Tensor`
        kwargs_conv1d (kwargs):
            Keyword arguments passed on to torch.nn.functional.conv1d
            (must not include `groups`, which is used for batching)
        """
        super().__init__()
        if not isinstance(fir, (list, np.ndarray, torch.Tensor)):
            raise TypeError(
                "fir must be list, np.ndarray or torch.Tensor, got "
                f"{fir.__class__.__name__}"
            )
        if isinstance(fir, (list, np.ndarray)):
            fir = torch.tensor(fir, dtype=dtype)
        if fir.ndim not in [1, 2]:
            raise ValueError(
                "fir must be one- or two-dimensional with shape (n_taps,) or "
                f"(n_filters, n_taps), got shape {fir.shape}"
            )
        self.register_buffer("fir", fir)
        self.kwargs_conv1d = kwargs_conv1d

    def forward(self, x, batching=False):
        """
        Filter input signal

        Args
        ----
        x (torch.Tensor): Input signal
        batching (bool):
            If `True`, the input is assumed to have shape (..., n_filters, time)
            and each channel is filtered with its own filter

        Returns
        -------
        y (torch.Tensor): Filtered signal
        """
        y = x
        if batching:
            assert y.shape[-2] == self.fir.shape[0]
        else:
            y = y.unsqueeze(-2)
        unflatten_shape = y.shape[:-2]
        y = torch.flatten(y, start_dim=0, end_dim=-2 - 1)
        y = torch.nn.functional.conv1d(
            input=torch.nn.functional.pad(y, (self.fir.shape[-1] - 1, 0)),
            weight=self.fir.flip(-1).view(-1, 1, self.fir.shape[-1]),
            **self.kwargs_conv1d,
            groups=y.shape[-2] if batching else 1,
        )
        y = torch.unflatten(y, 0, unflatten_shape)
        if self.fir.ndim == 1:
            y = y.squeeze(-2)
        return y


class GammatoneFilterbank(FIRFilterbank):
    def __init__(
        self,
        sr=20e3,
        fir_dur=0.05,
        cfs=utils.erbspace(8e1, 8e3, 50),
        dtype=torch.float32,
        **kwargs,
    ):
        """ """
        fir = filters.gammatone_filterbank_fir(
            sr=sr,
            fir_dur=fir_dur,
            cfs=cfs,
            **kwargs,
        )
        super().__init__(fir, dtype=dtype)


class IHCLowpassFilter(FIRFilterbank):
    def __init__(
        self,
        sr_input=20e3,
        sr_output=10e3,
        fir_dur=0.05,
        cutoff=3e3,
        order=7,
        dtype=torch.float32,
    ):
        """ """
        fir = filters.ihc_lowpass_filter_fir(
            sr=sr_input,
            fir_dur=fir_dur,
            cutoff=cutoff,
            order=order,
        )
        stride = int(sr_input / sr_output)
        msg = f"{sr_input=} and {sr_output=} require non-integer stride"
        assert np.isclose(stride, sr_input / sr_output), msg
        super().__init__(fir, dtype=dtype, stride=stride)


class HalfCosineFilterbank(torch.nn.Module):
    def __init__(
        self,
        sr=50000,
        cf_low=20e0,
        cf_high=20e3,
        cf_num=50,
        scale="erb",
        include_highpass=False,
        include_lowpass=False,
    ):
        """ """
        super().__init__()
        self.sr = sr
        self.cf_low = cf_low
        self.cf_high = cf_high
        self.cf_num = cf_num
        self.scale = scale
        self.include_lowpass = include_lowpass
        self.include_highpass = include_highpass
        self.cfs = None
        self.filters = None

    def half_cosine_transfer_function(
        self,
        f,
        cf,
        bw,
        lowpass=False,
        highpass=False,
    ):
        """
        Transfer function of a half-cosine filter with center frequency
        `cf` and bandwidth `bw`, evaluated at frequencies `f`.
        """
        out = np.zeros_like(f)
        IDX = np.logical_and(f > cf - bw / 2, f < cf + bw / 2)
        out[IDX] = np.cos(np.pi * (f[IDX] - cf) / bw)
        if lowpass:
            out[f < cf] = 1
        if highpass:
            out[f > cf] = 1
        return out

    def get_frequency_domain_filters(self, f):
        """
        Construct frequency domain half-cosine filterbank with transfer
        functions evaluted at frequencies `f`.
        """
        if self.scale.lower() == "erb":
            f = utils.freq2erb(f)
            cfs = np.linspace(
                start=utils.freq2erb(self.cf_low),
                stop=utils.freq2erb(self.cf_high),
                num=self.cf_num,
            )
            self.cfs = utils.erb2freq(cfs)
        elif self.scale.lower() == "log":
            f = np.log(f)
            cfs = np.linspace(
                start=np.log(self.cf_low),
                stop=np.log(self.cf_high),
                num=self.cf_num,
            )
            self.cfs = np.exp(cfs)
        elif self.scale.lower() == "linear":
            cfs = np.linspace(
                start=self.cf_low,
                stop=self.cf_high,
                num=self.cf_num,
            )
            self.cfs = cfs
        else:
            raise ValueError(f"unrecognized filterbank scale: {self.scale}")
        bw = 2 * (cfs[1] - cfs[0]) if self.cf_num > 1 else np.inf
        filters = np.zeros((self.cf_num, len(f)), dtype=f.dtype)
        for itr, cf in enumerate(cfs):
            filters[itr] = self.half_cosine_transfer_function(
                f=f,
                cf=cf,
                bw=bw,
                lowpass=(itr == 0) and (self.include_lowpass),
                highpass=(itr == self.cf_num - 1) and (self.include_highpass),
            )
        return filters

    def forward(self, x):
        """
        Apply filterbank along time axis (dim=1) in the frequency domain.
        Construct new filterbank on first call or if input shape changes.
        """
        assert x.ndim >= 2, "Expected input shape [batch, time, ...]"
        y = torch.fft.rfft(x, dim=1)
        y = torch.unsqueeze(y, dim=1)
        rebuild = self.filters is None
        rebuild = rebuild or (not y.device == self.filters.device)
        rebuild = rebuild or (not y.ndim == self.filters.ndim)
        rebuild = rebuild or (not y.shape[2] == self.filters.shape[2])
        if rebuild:
            f = np.fft.rfftfreq(x.shape[1], d=1 / self.sr)
            self.filters = torch.as_tensor(
                self.get_frequency_domain_filters(f),
                dtype=y.dtype,
                device=x.device,
            )
            self.filters = self.filters[None, ...]
            for _ in range(2, x.ndim):
                self.filters = self.filters[..., None]
        y = y * self.filters
        y = torch.fft.irfft(y, dim=2)
        return y


class GradientClippedPower(torch.autograd.Function):
    """
    Custom autograd Function for power function with gradient
    clipping (used to limit gradients for power compression).
    """

    @staticmethod
    def forward(ctx, input, power, clip_value):
        ctx.save_for_backward(input)
        ctx.power = power
        ctx.clip_value = clip_value
        return torch.pow(input, power)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad = ctx.power * torch.pow(input, ctx.power - 1)
        grad = torch.clamp(grad, min=None, max=ctx.clip_value)
        return grad_output * grad, None, None
