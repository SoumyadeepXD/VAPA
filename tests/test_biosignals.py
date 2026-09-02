"""
VAPA Subsystem Test 2: EMG & EEG Biosignal DSP and Intent Classifier
Tests real-time bio-signal acquisition, digital filters, envelope extraction,
and neural-muscular intent recognition with interactive gesture injection.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biosignals.biosignal_streamer import BiosignalStreamer
from biosignals.emg_decoder import EMGDecoder, EMGIntent
from biosignals.eeg_decoder import EEGDecoder, EEGIntent
from biosignals.intent_fusion import IntentFusionEngine


def run_biosignal_test(force_mock: bool = False):
    print("=" * 75)
    print(" VAPA TEST: Biosignals Pipeline (EMG Muscle & EEG Brain Wave Decoding)")
    print("=" * 75)

    streamer = BiosignalStreamer(force_mock=force_mock)
    emg_decoder = EMGDecoder()
    eeg_decoder = EEGDecoder()
    fusion = IntentFusionEngine()

    print("\nControls (Interactive Gesture Injection):")
    print("  'g' - Trigger Synthetic EMG Grasp Close (Flexor burst)")
    print("  'o' - Trigger Synthetic EMG Hand Open (Extensor burst)")
    print("  'e' - Trigger Synthetic EMG Co-Contraction E-STOP")
    print("  'r' - Trigger Synthetic EEG Motor Imagery Reach (Mu ERD)")
    print("  'c' - Trigger Synthetic EEG Target Cycle (Frontal blink spike)")
    print("  'q' - Quit test\n")

    # In a terminal loop, we can test automatic sweep or keyboard
    start_time = time.time()
    last_print = time.time()

    # If non-blocking stdin is tricky in pure terminal, we simulate sequential triggers or accept input
    print("Starting real-time signal processing loop (100 Hz)...\n")

    try:
        sample_step = 0
        while True:
            t0 = time.time()
            emg_chunk, eeg_chunk = streamer.read_chunk(num_samples=10)

            emg_intent = emg_decoder.update_samples(emg_chunk)
            eeg_intent = eeg_decoder.update_samples(eeg_chunk)
            cmd = fusion.fuse(emg_intent, eeg_intent, [], "SCANNING")

            # Periodic automatic trigger injection for demonstration
            sample_step += 1
            if streamer.is_synthetic and sample_step % 250 == 0:
                actions = ["grasp", "open", "reach", "cycle"]
                act = actions[(sample_step // 250) % len(actions)]
                streamer.trigger_synthetic_gesture(act, duration_s=0.6)
                print(f"--> [AUTO-TRIGGER] Injected synthetic '{act}' event.")

            now = time.time()
            if now - last_print >= 0.25:  # Print telemetry at 4 Hz
                last_print = now

                # Format EMG Bar
                bar_len = 20
                emg_filled = int(bar_len * emg_intent.activation_level)
                emg_bar = "[" + "#" * emg_filled + "-" * (bar_len - emg_filled) + "]"

                # Format EEG Bar
                eeg_filled = int(bar_len * eeg_intent.attention_score)
                eeg_bar = "[" + "=" * eeg_filled + " " * (bar_len - eeg_filled) + "]"

                sys.stdout.write(
                    f"\rEMG: {emg_intent.gesture:20s} {emg_bar} Act: {emg_intent.activation_level:.2f} | "
                    f"EEG: {eeg_intent.command:18s} {eeg_bar} Mu: {eeg_intent.mu_power:.2f} | "
                    f"FUSION: {cmd.action:14s}"
                )
                sys.stdout.flush()

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    finally:
        streamer.close()
        print("\nBiosignals Test Finished.")


if __name__ == "__main__":
    force_mock_flag = "--mock" in sys.argv or "--real" not in sys.argv
    run_biosignal_test(force_mock=force_mock_flag)
