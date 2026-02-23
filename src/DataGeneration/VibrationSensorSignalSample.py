import numpy as np
from scipy.interpolate import interp1d


class VibrationSensorSignalSample:
    CUTOFF: int = 150

    def __init__(self, w, a, fundamental_from, fundamental_to, health_index=1.0,
                 t=0, interval=1, previous_sample=None, sample_rate=1024, fault_type="normal"):
        self.__last_cumsum = None
        self.interval = interval
        self.sample_rate = sample_rate
        self.W = w
        self.A = a
        self.t = t
        self.health_index = health_index
        self.fault_type = fault_type
        self.base_frequency = fundamental_from
        self.target_base_frequency = fundamental_to
        self.add_noise = True
        self.__previous_sample = previous_sample
        self.__N = int(sample_rate * interval)

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

        # 2. Emergent Non-linear Harmonics (Existing Logic)
        wear_severity = max(0.0, 1.0 - self.health_index)
        if self.health_index < 0.8:
            fault_frequencies = [13.7, 21.3, 44.5]
            fault_amplitudes = [5.0 * wear_severity, 8.0 * wear_severity, 12.0 * wear_severity]
            for fw, fa in zip(fault_frequencies, fault_amplitudes):
                signal += np.sin(base * fw) * fa

        # 3. IMPACT SHOCKS (New Logic for Bearing Faults)
        # Shocks occur if the bearing fault is active and the machine is spinning
        if self.fault_type == "bearing" and self.health_index < 0.7 and np.mean(f) > 1:
            # BPFO (Ball Pass Frequency Outer) is typically ~3x to 10x shaft speed
            impact_freq_factor = 4.5
            impact_points = np.where(np.diff(np.floor(fi * impact_freq_factor)) > 0)[0]

            # Create a decaying exponential shock pulse for each impact point
            pulse_len = int(self.sample_rate * 0.02)  # 20ms pulse
            shock_magnitude = 40.0 * wear_severity

            for idx in impact_points:
                if idx + pulse_len < self.__N:
                    # Decaying sine wave: sin(low_freq) * exp(-decay)
                    time_window = np.linspace(0, 0.02, pulse_len)
                    # High-frequency "ring-down" of the machine structure
                    shock_pulse = np.sin(2 * np.pi * 200 * time_window) * np.exp(-200 * time_window)
                    signal[idx:idx + pulse_len] += shock_pulse * shock_magnitude

        # 4. Dynamic Noise Floor
        noise_magnitude = 0.1 + wear_severity * 2.0
        if self.add_noise:
            signal += np.random.normal(0, noise_magnitude, self.__N)

        self.__last_cumsum = fi[-1]

        signal[signal > self.CUTOFF] = self.CUTOFF
        signal[signal < -self.CUTOFF] = -self.CUTOFF

        return np.int16(signal / self.CUTOFF * 32767)
