"""
VAPA Digital Signal Processing (DSP) & Bio-Filter Suite
Filters raw microvolt-level EMG/EEG signals: Notch filtering (50/60Hz),
Bandpass filtering, Rectification, Moving Envelope, and Spectral Analysis.
"""

import math
import numpy as np
try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class SignalFilter:
    """Real-time multi-channel IIR filter with state memory."""
    def __init__(self, sampling_rate_hz: float):
        self.fs = float(sampling_rate_hz)
        self.filters = {}

    def add_notch_filter(self, name: str, notch_freq_hz: float = 50.0, q: float = 30.0):
        """Adds a powerline notch filter (50Hz or 60Hz)."""
        if SCIPY_AVAILABLE:
            b, a = signal.iirnotch(notch_freq_hz, q, self.fs)
            self.filters[name] = {"b": b, "a": a, "type": "iir"}
        else:
            self.filters[name] = {"type": "passthrough"}

    def add_bandpass_filter(self, name: str, low_hz: float, high_hz: float, order: int = 4):
        """Adds a Butterworth bandpass filter."""
        if SCIPY_AVAILABLE:
            nyq = 0.5 * self.fs
            low = max(0.001, low_hz / nyq)
            high = min(0.999, high_hz / nyq)
            b, a = signal.butter(order, [low, high], btype="bandpass")
            self.filters[name] = {"b": b, "a": a, "type": "iir"}
        else:
            self.filters[name] = {"type": "passthrough"}

    def filter_signal(self, data: np.ndarray, filter_name: str) -> np.ndarray:
        """Applies specified filter along the last axis of data."""
        if filter_name not in self.filters:
            return data

        flt = self.filters[filter_name]
        if flt["type"] == "iir" and SCIPY_AVAILABLE:
            if data.ndim == 1:
                return signal.lfilter(flt["b"], flt["a"], data)
            else:
                return signal.lfilter(flt["b"], flt["a"], data, axis=-1)
        return data


def compute_rms_envelope(raw_signal: np.ndarray, window_size: int = 50) -> np.ndarray:
    """
    Computes moving Root Mean Square (RMS) envelope of the EMG signal.
    window_size: number of samples over which RMS is computed.
    """
    squared = np.square(raw_signal)
    if squared.ndim == 1:
        window = np.ones(window_size) / window_size
        rms = np.sqrt(np.convolve(squared, window, mode="same"))
    else:
        # Multi-channel array (channels x samples)
        rms = np.zeros_like(raw_signal)
        window = np.ones(window_size) / window_size
        for ch in range(raw_signal.shape[0]):
            rms[ch] = np.sqrt(np.convolve(squared[ch], window, mode="same"))
    return rms


def compute_mav(raw_signal: np.ndarray) -> float:
    """Mean Absolute Value (MAV) of a signal window."""
    return float(np.mean(np.abs(raw_signal)))


def compute_waveform_length(raw_signal: np.ndarray) -> float:
    """Waveform Length (WL): Cumulative length of the waveform over the time segment."""
    return float(np.sum(np.abs(np.diff(raw_signal))))


def compute_bandpower(raw_signal: np.ndarray, sampling_rate_hz: float, band: tuple[float, float]) -> float:
    """
    Calculates average power in a specific frequency band [f_low, f_high] via FFT.
    Used for EEG Mu (8-12Hz) and Beta (13-30Hz) rhythms.
    """
    n = len(raw_signal)
    if n < 8:
        return 0.0

    # Compute FFT
    freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate_hz)
    psd = np.abs(np.fft.rfft(raw_signal)) ** 2 / n

    # Find frequency indices in target band
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    band_power = np.sum(psd[idx_band])
    total_power = np.sum(psd) + 1e-10

    # Return relative band power (0.0 to 1.0)
    return float(band_power / total_power)
