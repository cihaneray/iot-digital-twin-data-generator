import numpy as np
import matplotlib.pyplot as plt
from DataGeneration import VibrationSensorSignalSample


def diagnostic_plot():
    # Parameters
    sample_rate = 2048  # Higher rate to capture the 200Hz shocks clearly
    duration = 0.5  # Half a second is enough to see pulses
    W = [1, 2, 3]  # Base harmonics
    A = [15, 8, 4]  # Amplitudes
    freq = 25  # 25 Hz shaft speed (1500 RPM)

    # 1. Healthy Signal (No shocks, low noise)
    healthy_gen = VibrationSensorSignalSample(
        w=W, a=A, fundamental_from=freq, fundamental_to=freq,
        health_index=1.0, fault_type="normal",
        sample_rate=sample_rate, interval=duration
    )
    healthy_pcm = healthy_gen.pcm()

    # 2. Bearing Fault Signal (Shocks + high noise)
    failing_gen = VibrationSensorSignalSample(
        w=W, a=A, fundamental_from=freq, fundamental_to=freq,
        health_index=0.4, fault_type="bearing",
        sample_rate=sample_rate, interval=duration
    )
    failing_pcm = failing_gen.pcm()

    # Time axis
    t = np.linspace(0, duration, len(healthy_pcm))

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax1.plot(t, healthy_pcm, color='blue', alpha=0.7)
    ax1.set_title('Healthy Machine Vibration (Standard Harmonics)')
    ax1.set_ylabel('Amplitude (PCM)')
    ax1.grid(True)

    ax2.plot(t, failing_pcm, color='red', alpha=0.8)
    ax2.set_title('Bearing Fault Vibration (Impact Shocks & Decaying Ring-down)')
    ax2.set_ylabel('Amplitude (PCM)')
    ax2.set_xlabel('Time (seconds)')
    ax2.grid(True)

    # Highlight a few impact zones for the user
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    diagnostic_plot()