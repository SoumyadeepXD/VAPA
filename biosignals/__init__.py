"""
VAPA Biosignal Processing & Neural Decoding Package (EMG & EEG)
"""
from biosignals.signal_filters import SignalFilter, compute_rms_envelope, compute_bandpower
from biosignals.emg_decoder import EMGDecoder, EMGIntent
from biosignals.eeg_decoder import EEGDecoder, EEGIntent
from biosignals.biosignal_streamer import BiosignalStreamer, SyntheticBiosignalStreamer
from biosignals.intent_fusion import IntentFusionEngine, MultimodalCommand
