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
            active_faults: set[str] | frozenset[str] = frozenset(),
            speed_ratio: float = 0.0,
            load_ratio: float = 0.0,
            fault_activation_health_threshold: float = 0.85,
            vibration_full_signature_health: float = 0.80,
            fault_vibration_speed_exponent: float = 1.5,
            fault_vibration_min_speed_ratio: float = 0.02,
            fault_vibration_load_influence: float = 0.5,
            fault_vibration_health_exponent: float = 1.0,
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
        self.active_faults = active_faults
        self.speed_ratio = speed_ratio
        self.load_ratio = load_ratio
        self.fault_activation_health_threshold = fault_activation_health_threshold
        self.vibration_full_signature_health = vibration_full_signature_health
        self.fault_vibration_speed_exponent = fault_vibration_speed_exponent
        self.fault_vibration_min_speed_ratio = fault_vibration_min_speed_ratio
        self.fault_vibration_load_influence = fault_vibration_load_influence
        self.fault_vibration_health_exponent = fault_vibration_health_exponent
        self.base_frequency = fundamental_from
        self.target_base_frequency = fundamental_to
        self.add_noise = True
        self._N = int(sample_rate * interval)
        self._rng: np.random.Generator = rng if rng is not None else np.random.default_rng()

    def _fault_gain(self, wear_severity: float, signature_ramp: float) -> float:
        speed_base = max(self.fault_vibration_min_speed_ratio, self.speed_ratio)
        speed_gain = speed_base ** self.fault_vibration_speed_exponent

        load_clamped = float(np.clip(self.load_ratio, 0.0, 1.5))
        load_gain = (
            (1.0 - self.fault_vibration_load_influence)
            + self.fault_vibration_load_influence * load_clamped
        )

        if wear_severity <= 0.0:
            health_gain = 0.0
        else:
            health_gain = wear_severity ** self.fault_vibration_health_exponent

        return speed_gain * load_gain * health_gain * signature_ramp

    @property
    def last_cumsum(self) -> float:
        if self._last_cumsum is None:
            raise RuntimeError(
                "last_cumsum is not available until pcm() has been called."
            )
        return self._last_cumsum

    def pcm(self) -> np.signedinteger:
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

        ramp_start = self.fault_activation_health_threshold
        ramp_end = self.vibration_full_signature_health
        if ramp_start > ramp_end:
            signature_ramp = (ramp_start - self.health_index) / (ramp_start - ramp_end)
            signature_ramp = float(np.clip(signature_ramp, 0.0, 1.0))
        else:
            signature_ramp = 1.0

        fault_gain = self._fault_gain(wear_severity, signature_ramp)

        if "imbalance" in self.active_faults and self.health_index < ramp_start:
            for order, amp in zip(self.IMBALANCE_ORDERS, self.IMBALANCE_AMPLITUDES):
                signal += np.sin(base * order) * amp * fault_gain

        if "bearing" in self.active_faults and self.health_index < ramp_start:
            for order, amp in zip(self.BEARING_ORDERS, self.BEARING_AMPLITUDES):
                signal += np.sin(base * order) * amp * fault_gain

        if (
                "bearing" in self.active_faults
                and self.health_index < 0.7
                and np.mean(f) > 1.0
        ):
            impact_freq_factor = self.BEARING_ORDERS[1]  # 3.5
            impact_points = np.where(np.diff(np.floor(fi * impact_freq_factor)) > 0)[0]
            impact_points = impact_points + 1  # Correct index alignment from diff

            shock_magnitude = 40.0 * fault_gain
            max_pulse_len = int(self.sample_rate * 0.02)  # 20 ms ring-down max

            for i, idx in enumerate(impact_points):
                if idx >= self._N:
                    continue

                # Determine when the current pulse must yield to the next
                if i < len(impact_points) - 1:
                    next_idx = impact_points[i + 1]
                    pulse_len = min(max_pulse_len, next_idx - idx)
                else:
                    pulse_len = max_pulse_len

                # Avoid overflowing the signal array
                pulse_len = min(pulse_len, self._N - idx)
                
                if pulse_len <= 0:
                    continue

                actual_duration = pulse_len / self.sample_rate
                time_window = np.linspace(0, actual_duration, pulse_len, endpoint=False)
                shock_pulse = (
                        np.sin(2 * np.pi * 200 * time_window)
                        * np.exp(-200 * time_window)
                )
                signal[idx:idx + pulse_len] += shock_pulse * shock_magnitude

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
