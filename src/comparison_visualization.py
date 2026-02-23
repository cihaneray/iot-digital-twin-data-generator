import numpy as np
import matplotlib.pyplot as plt
from DataGeneration import VibrationSensorSignalSample


def visualize_fft_comparison():
    # Simulation Parameters
    sample_rate = 1024
    duration = 2  # seconds
    W = [1, 2, 3]  # Fundamental and first two harmonics
    A = [10, 5, 2]  # Amplitudes
    freq = 30  # 30 Hz (1800 RPM)

    # 1. Generate Healthy Signal (Health = 1.0)
    healthy_gen = VibrationSensorSignalSample(
        w=W, a=A, fundamental_from=freq, fundamental_to=freq,
        health_index=1.0, sample_rate=sample_rate, interval=duration
    )
    healthy_pcm = healthy_gen.pcm()

    # 2. Generate Failing Signal (Health = 0.2)
    failing_gen = VibrationSensorSignalSample(
        w=W, a=A, fundamental_from=freq, fundamental_to=freq,
        health_index=0.2, sample_rate=sample_rate, interval=duration
    )
    failing_pcm = failing_gen.pcm()

    # FFT Calculation
    n = len(healthy_pcm)
    freq_axis = np.fft.rfftfreq(n, d=1 / sample_rate)

    healthy_fft = np.abs(np.fft.rfft(healthy_pcm)) / n
    failing_fft = np.abs(np.fft.rfft(failing_pcm)) / n

    # Plotting
    plt.figure(figsize=(12, 6))

    plt.plot(freq_axis, healthy_fft, label='Healthy (1.0)', alpha=0.8, color='blue')
    plt.plot(freq_axis, failing_fft, label='Failing (0.2)', alpha=0.6, color='red')

    plt.title('Vibration Spectrum: Healthy vs. Failing Machine')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.yscale('log')  # Log scale highlights emergent low-amplitude harmonics
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.xlim(0, 250)

    plt.show()


if __name__ == "__main__":
    visualize_fft_comparison()
