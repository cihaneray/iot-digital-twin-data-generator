import numpy as np
from scipy.interpolate import interp1d


class VibrationSensorSignalSample:
    CUTOFF: int = 150

    def __init__(self, w, a, fundamental_from, fundamental_to, health_index=1.0,
                 t=0, interval: int = 1, previous_sample=None, sample_rate: int = 1024):
        self.__last_cumsum = None
        self.interval = interval
        self.sample_rate = sample_rate
        self.W = w
        self.A = a
        self.t = t
        self.health_index = health_index
        self.base_frequency = fundamental_from
        self.target_base_frequency = fundamental_to
        self.add_noise = True
        self.__previous_sample = previous_sample
        self.__N = sample_rate * interval

    def pcm(self):
        ts = np.linspace(self.t, self.t + self.interval, num=self.__N, endpoint=False)

        x = np.array([0, self.interval]) + self.t
        points = np.array([self.base_frequency, self.target_base_frequency])
        rpm = interp1d(x, points, kind='linear')

        f = rpm(ts)
        f[f < 0] = 0

        fi = np.cumsum(f / self.sample_rate) + (self.__previous_sample.__last_cumsum if self.__previous_sample else 0)
        base = 2 * np.pi * fi

        # 1. Standard Harmonics
        signal = np.array([np.sin(base * w) * a for w, a in zip(self.W, self.A)]).sum(axis=0)

        # 2. Emergent Non-linear Harmonics (Fault Logic)
        # As health_index drops, we add high-frequency rattling (e.g., 20x and 50x fundamental)
        if self.health_index < 0.8:
            wear_severity = (1.0 - self.health_index)

            # Noninteger harmonics simulate asynchronous vibration or loose parts
            fault_frequencies = [13.7, 21.3, 44.5]
            fault_amplitudes = [5.0 * wear_severity, 8.0 * wear_severity, 12.0 * wear_severity]

            for fw, fa in zip(fault_frequencies, fault_amplitudes):
                signal += np.sin(base * fw) * fa

        # 3. Dynamic Noise Floor
        # Failed machines are "louder" across the entire spectrum
        noise_magnitude = 0.1 + (1.0 - self.health_index) * 2.0
        if self.add_noise:
            signal += np.random.normal(0, noise_magnitude, self.__N)

        self.__last_cumsum = fi[-1]

        # Clipping and scaling
        signal[signal > self.CUTOFF] = self.CUTOFF
        signal[signal < -self.CUTOFF] = -self.CUTOFF

        return np.int16(signal / self.CUTOFF * 32767)
