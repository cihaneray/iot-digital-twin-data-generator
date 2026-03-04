import numpy as np
from scipy.interpolate import interp1d


class VibrationSensorSignalSample:
    """
    Generates a synthetic vibration PCM signal for a rotational machine.

    Phase continuity across consecutive calls is maintained via `prev_cumsum`,
    a plain float passed in from the caller. This avoids building a linked-list
    chain of sample objects (which caused unbounded memory growth in the old design).

    Call `pcm()` to compute the signal. `last_cumsum` is populated after that call
    and should be stored by the caller for the next sample.
    """

    CUTOFF: int = 150

    # Characteristic frequency multipliers (dimensionless orders of fundamental)
    # Imbalance:  dominant at 1× and 2× running speed
    IMBALANCE_ORDERS = [1.0, 2.0]
    IMBALANCE_AMPLITUDES = [8.0, 3.0]

    # Ball-pass frequency outer race (BPFO) ≈ 3.5×, inner (BPFI) ≈ 5.4×,
    # ball spin frequency (BSF) ≈ 2.3× — representative values for a 6-ball bearing.
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
        """
        The final phase accumulator value after signal generation.
        Raises RuntimeError if accessed before pcm() has been called.
        """
        if self._last_cumsum is None:
            raise RuntimeError(
                "last_cumsum is not available until pcm() has been called."
            )
        return self._last_cumsum

    def pcm(self) -> np.ndarray:
        """
        Compute and return the int16 PCM vibration signal.
        Must be called before accessing `last_cumsum`.
        """
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
            # Impact repetition rate uses BPFO order (3.5×) as the dominant defect frequency
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

        np.clip(signal, -self.CUTOFF, self.CUTOFF, out=signal)
        pcm_signal = np.int16(signal / self.CUTOFF * 32767)

        self._last_cumsum = float(fi[-1])

        return pcm_signal
