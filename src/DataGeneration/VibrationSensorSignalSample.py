import numpy as np
from scipy.interpolate import interp1d


class VibrationSensorSignalSample:
    CUTOFF: int = 150

    IMBALANCE_ORDERS = [1.0, 2.0]
    IMBALANCE_AMPLITUDES = [8.0, 3.0]

    BEARING_ORDERS = [2.3, 3.5, 5.4]
    BEARING_AMPLITUDES = [5.0, 8.0, 12.0]

    def __init__(
            self,
            w: list,
            a: list,
            fundamental_from: float,
            fundamental_to: float,
            health_index: float = 1.0,
            t: int = 0,
            interval: float = 1.0,
            prev_cumsum: float = 0.0,
            sample_rate: int = 1024,
            fault_type: str = "normal",
            rng: np.random.Generator | None = None,
    ):
        self._last_cumsum: float | None = None
        self._prev_cumsum = prev_cumsum

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
        self._N = int(sample_rate * interval)
        self._rng: np.random.Generator = rng if rng is not None else np.random.default_rng()

    @property
    def last_cumsum(self) -> float:
        if self._last_cumsum is None:
            raise RuntimeError(
                "last_cumsum is not available until pcm() has been called."
            )
        return self._last_cumsum

    def pcm(self) -> np.ndarray:
        ts = np.linspace(self.t, self.t + self.interval, num=self._N, endpoint=False)

        x = np.array([0, self.interval]) + self.t
        points = np.array([self.base_frequency, self.target_base_frequency])
        rpm_interp = interp1d(x, points, kind="linear")

        f = rpm_interp(ts)
        f[f < 0] = 0.0

        fi = np.cumsum(f / self.sample_rate) + self._prev_cumsum
        base = 2 * np.pi * fi

        signal = np.sum(
            [np.sin(base * w) * a for w, a in zip(self.W, self.A)], axis=0
        )

        wear_severity = max(0.0, 1.0 - self.health_index)

        if self.fault_type == "imbalance" and self.health_index < 0.8:
            for order, amp in zip(self.IMBALANCE_ORDERS, self.IMBALANCE_AMPLITUDES):
                signal += np.sin(base * order) * amp * wear_severity

        if self.fault_type == "bearing" and self.health_index < 0.8:
            for order, amp in zip(self.BEARING_ORDERS, self.BEARING_AMPLITUDES):
                signal += np.sin(base * order) * amp * wear_severity

        if (
                self.fault_type == "bearing"
                and self.health_index < 0.7
                and np.mean(f) > 1.0
        ):
            impact_freq_factor = self.BEARING_ORDERS[1]  # 3.5
            impact_points = np.where(np.diff(np.floor(fi * impact_freq_factor)) > 0)[0]

            pulse_len = int(self.sample_rate * 0.02)  # 20 ms ring-down
            shock_magnitude = 40.0 * wear_severity

            for idx in impact_points:
                end = idx + pulse_len
                if end < self._N:
                    time_window = np.linspace(0, 0.02, pulse_len)
                    shock_pulse = (
                            np.sin(2 * np.pi * 200 * time_window)
                            * np.exp(-200 * time_window)
                    )
                    signal[idx:end] += shock_pulse * shock_magnitude

        if self.add_noise:
            noise_magnitude = 0.1 + wear_severity * 2.0
            signal += self._rng.normal(0, noise_magnitude, self._N)

        # Soft Saturation (tanh) to simulate sensor non-linearity
        # Instead of hard clipping at CUTOFF, we use tanh to compress the signal.
        # This prevents "square wave" artifacts while keeping the signal within bounds.
        scaled_signal = signal / (self.CUTOFF * 0.8)  # Leave some headroom before saturation
        saturated_signal = np.tanh(scaled_signal) * self.CUTOFF

        pcm_signal = np.int16(saturated_signal / self.CUTOFF * 32767)

        self._last_cumsum = float(fi[-1])

        return pcm_signal
