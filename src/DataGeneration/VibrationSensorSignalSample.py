import numpy as np
from scipy.interpolate import interp1d


class VibrationSensorSignalSample:
    CUTOFF: int = 150

    def __init__(self, w, a, fundamental_from, fundamental_to, t=0, interval=1, previous_sample=None, sample_rate=1024):
        self.__last_cumsum = None
        self.interval = interval
        self.sample_rate = sample_rate
        self.W = w
        self.A = a
        self.t = t
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
        b = np.array([np.sin(base * w) * a for w, a in zip(self.W, self.A)])
        a = b.sum(axis=0)

        if self.add_noise:
            a += np.random.normal(0, 0.1, self.__N)

        self.__last_cumsum = fi[-1]
        self.base_frequency = self.target_base_frequency

        a[a > self.CUTOFF] = self.CUTOFF
        a[a < -self.CUTOFF] = -self.CUTOFF

        return np.int16(a / self.CUTOFF * 32767)
