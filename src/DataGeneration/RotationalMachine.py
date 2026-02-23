import math
import random

from .VibrationSensorSignalSample import VibrationSensorSignalSample


class RotationalMachine:
    AMBIENT_TEMPERATURE: float = 20.0  # degrees Celsius
    MAX_TEMPERATURE: float = 120.0
    AMBIENT_PRESSURE: float = 101.0  # kPa
    FAILURE_THRESHOLD: float = 100.0  # Damage threshold where failure occurs

    def __init__(self, name):
        self.name = name
        self.W = [1 / 2, 1, 2, 3, 5, 7, 12, 18]
        self.A = [1, 5, 80, 2 / 3, 8, 2, 14, 50]
        self.t = 0
        self.speed = 0.0
        self.speed_desired = 0.0
        self.temperature = self.AMBIENT_TEMPERATURE
        self.pressure = self.AMBIENT_PRESSURE

        # State-based health variables
        self.accumulated_damage = 0.0
        self.broken = False
        self.__vibration_sample = None

    @property
    def health_index(self):
        """Returns a health factor from 1.0 (new) to 0.0 (failed)."""
        return max(0.0, 1.0 - (self.accumulated_damage / self.FAILURE_THRESHOLD))

    def set_speed(self, speed):
        self.speed_desired = speed

    def calculate_wear_delta(self):
        """Calculates damage increment based on current physics."""
        # Base wear per simulation step
        base_wear = 0.001

        # Speed stress: wear increases quadratically with speed
        speed_factor = (self.speed / 1000) ** 2

        # Thermal stress: wear increases exponentially if above 80 degrees
        temp_factor = math.exp((self.temperature - 80) / 20) if self.temperature > 80 else 1.0

        return base_wear * speed_factor * temp_factor

    def next_state(self):
        if self.broken:
            raise Exception("Machine is in a failed state. Repair required.")

        # 1. Accumulate Damage
        damage_step = self.calculate_wear_delta()
        self.accumulated_damage += damage_step

        h = self.health_index
        if h <= 0:
            self.broken = True
            raise Exception("Mechanical Failure: Failure Threshold Reached")

        # 2. Physics updates influenced by health (h)
        # As health decreases (h -> 0), friction increases temperature and reduces speed efficiency
        v_from = self.speed / 60

        # Lower health makes it harder to reach the desired speed
        efficiency = 0.5 + (0.5 * h)
        self.speed = (self.speed + (efficiency * self.speed_desired)) / 2
        v_to = self.speed / 60

        # Thermal dynamics with health-induced friction
        target_temp = (self.speed / 10) * (2 - h)  # Degraded machines run hotter
        temp_gain_rate = 0.01 * (self.speed / 1000)
        self.temperature += (target_temp - self.temperature) * temp_gain_rate

        # Pressure increases as health declines (internal resistance)
        target_pressure = (self.speed * 2) / max(h, 0.1)
        self.pressure += (target_pressure - self.pressure) * 0.1

        # 3. Vibration - Increased noise and amplitude as health drops
        vibration_health_modifier = 1.0 + (1.0 - h) * 5
        modified_A = [amp * vibration_health_modifier for amp in self.A]

        self.__vibration_sample = VibrationSensorSignalSample(
            self.W, modified_A, v_from, v_to, t=self.t)

        state = {
            'speed_desired': self.speed_desired,
            'ambient_temperature': self.AMBIENT_TEMPERATURE + random.uniform(-0.1, 0.1),
            'speed': self.speed + random.uniform(-5, 5),
            'temperature': self.temperature + random.uniform(-0.1, 0.1),
            'pressure': self.pressure + random.uniform(-20, 20),
            'health_index': round(h, 4)
        }

        self.t += 1
        return {k: round(v, 2) if isinstance(v, (int, float)) else v for k, v in state.items()}
