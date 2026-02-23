import math
import random

from .Types import GlobalSimulationConfig
from .VibrationSensorSignalSample import VibrationSensorSignalSample


class RotationalMachine:
    AMBIENT_TEMPERATURE: float = 20.0  # degrees Celsius
    MAX_TEMPERATURE: float = 120.0
    AMBIENT_PRESSURE: float = 101.0  # kPa

    def __init__(self, name, config: 'GlobalSimulationConfig'):
        self.name = name
        self.config = config  # Reference to global stress parameters
        self.W = [1 / 2, 1, 2, 3, 5, 7, 12, 18]
        self.A = [1, 5, 80, 2 / 3, 8, 2, 14, 50]
        self.t = 0
        self.speed = 0.0
        self.speed_desired = 0.0
        self.temperature = self.AMBIENT_TEMPERATURE
        self.pressure = self.AMBIENT_PRESSURE
        self.accumulated_damage = 0.0
        self.broken = False

    @property
    def health_index(self):
        return max(0.0, 1.0 - (self.accumulated_damage / self.config.failure_threshold))

    def set_speed(self, speed):
        """Sets the target velocity for the machine's speed controller."""
        self.speed_desired = speed

    def calculate_wear_delta(self):
        # Base wear modified by global stress multiplier
        base_wear = 0.001 * self.config.stress_multiplier

        speed_factor = (self.speed / 1000) ** 2
        temp_factor = math.exp((self.temperature - 80) / 20) if self.temperature > 80 else 1.0

        return base_wear * speed_factor * temp_factor

    def next_state(self):
        if self.broken:
            raise Exception("Mechanical Failure")

        self.accumulated_damage += self.calculate_wear_delta()
        h = self.health_index

        if h <= 0:
            self.broken = True
            raise Exception("Mechanical Failure")

        # Physics simulation influenced by health
        v_from = self.speed / 60
        efficiency = 0.5 + (0.5 * h)
        self.speed = (self.speed + (efficiency * self.speed_desired)) / 2
        v_to = self.speed / 60

        self.temperature += ((self.speed / 10) * (2 - h) - self.temperature) * (0.01 * self.speed / 1000)
        self.pressure += ((self.speed * 2) / max(h, 0.1) - self.pressure) * 0.1

        # Vibration generation
        vibration_health_modifier = 1.0 + (1.0 - h) * 5
        modified_A = [amp * vibration_health_modifier for amp in self.A]

        sample = VibrationSensorSignalSample(
            self.W, modified_A, v_from, v_to, health_index=h, t=self.t
        )

        self.t += 1
        return {
            'speed_desired': round(self.speed_desired, 2),
            'speed': round(self.speed + random.uniform(-5, 5), 2),
            'temperature': round(self.temperature + random.uniform(-0.1, 0.1), 2),
            'pressure': round(self.pressure + random.uniform(-20, 20), 2),
            'health_index': round(h, 4),
            'vibration': sample
        }
