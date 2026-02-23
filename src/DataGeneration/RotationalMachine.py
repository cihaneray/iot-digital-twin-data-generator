import random

import numpy as np

from .VibrationSensorSignalSample import VibrationSensorSignalSample


class RotationalMachine:
    ambient_temperature: float = 20.0  # degrees Celsius
    max_temperature: float = 120.0
    ambient_pressure: float = 101.0  # kPa

    def __init__(self, name, h1, h2):
        self.W = [1 / 2, 1, 2, 3, 5, 7, 12, 18]
        self.A = [1, 5, 80, 2 / 3, 8, 2, 14, 50]
        self.t = 0
        self.name = name
        self.speed = 0
        self.speed_desired = 0
        self.temperature = RotationalMachine.ambient_temperature
        self.pressure = RotationalMachine.ambient_pressure
        self.pressure_factor = 2
        self.__vibration_sample = None
        self.__h1 = h1
        self.__h2 = h2
        self.broken = False

    def set_health(self, h1, h2):
        self.__h1 = h1
        self.__h2 = h2
        self.broken = False

    def set_speed(self, speed):
        self.speed_desired = speed

    @staticmethod
    def __g(v, min_v, max_v, target, rate):
        delta = (target - v) * rate
        return max(min(v + delta, max_v), min_v)

    @staticmethod
    def noise(magnitude):
        return random.uniform(-magnitude, magnitude)

    def next_state(self):
        try:
            _, h1 = next(self.__h1)
        except:
            self.broken = True
            raise Exception("F1")

        try:
            _, h2 = next(self.__h2)
        except:
            self.broken = True
            raise Exception("F2")

        v_from = self.speed / 60
        self.speed = (self.speed + (2 - h2) * self.speed_desired) / 2
        v_to = self.speed / 60

        self.temperature = (2 - h1) * self.__g(self.temperature, self.ambient_temperature, self.max_temperature,
                                               self.speed / 10, 0.01 * self.speed / 1000)
        self.pressure = h1 * self.__g(self.pressure, self.ambient_pressure, np.inf, self.speed * self.pressure_factor,
                                      0.3 * self.speed / 1000)
        self.__vibration_sample = VibrationSensorSignalSample(
            # self.W, self.A, v_from, v_to, t = self.t, previous_sample = self.__vibration_sample)
            self.W, self.A, v_from, v_to, t=self.t)

        state = {
            'speed_desired': self.speed_desired,
            'ambient_temperature': self.ambient_temperature + self.noise(0.1),
            'ambient_pressure': self.ambient_pressure + self.noise(0.1),
            'speed': self.speed + self.noise(5),
            'temperature': self.temperature + self.noise(0.1),
            'pressure': self.pressure + self.noise(20),
            'vibration': self.__vibration_sample
        }

        self.t += 1

        for key in state:
            value = state[key]
            if isinstance(value, (int, float)):
                state[key] = round(value, 2)

        return state
