import math
import random

from .Types import GlobalSimulationConfig
from .VibrationSensorSignalSample import VibrationSensorSignalSample


class RotationalMachine:
    MAX_TEMPERATURE: float = 120.0
    AMBIENT_PRESSURE: float = 101.0

    def __init__(self, name, config: 'GlobalSimulationConfig'):
        self.name = name
        self.config = config
        self.W = [1 / 2, 1, 2, 3, 5, 7, 12, 18]
        self.A = [1, 5, 80, 2 / 3, 8, 2, 14, 50]
        self.t = 0
        self.speed = 0.0
        self.speed_desired = 0.0
        self.ambient_temperature = self.config.ambient_temp_mean
        self.temperature = self.ambient_temperature
        self.pressure = self.AMBIENT_PRESSURE
        self.accumulated_damage = 0.0
        self.broken = False
        self.fault_type = self._assign_fault_type()
        self.fault_active = False

    def _assign_fault_type(self):
        r = random.random()
        if r < self.config.imbalance_probability: return "imbalance"
        if r < (self.config.imbalance_probability + self.config.bearing_fault_probability): return "bearing"
        return "normal"

    @property
    def health_index(self):
        return max(0.0, 1.0 - (self.accumulated_damage / self.config.failure_threshold))

    def set_speed(self, speed):
        self.speed_desired = speed

    def calculate_wear_delta(self):
        base_wear = 0.001 * self.config.stress_multiplier
        fault_multiplier = 1.0
        if self.fault_active:
            if self.fault_type == "bearing":
                fault_multiplier = 1.2
            elif self.fault_type == "imbalance":
                fault_multiplier = 1.1

        speed_factor = (self.speed / 1000) ** 2
        temp_factor = math.exp((self.temperature - 80) / 20) if self.temperature > 80 else 1.0
        return base_wear * speed_factor * temp_factor * fault_multiplier

    def next_state(self):
        if self.broken:
            raise Exception(f"Mechanical Failure: {self.fault_type}")

        if self.health_index < 0.85:
            self.fault_active = True

        self.accumulated_damage += self.calculate_wear_delta()
        h = self.health_index

        if h <= 0:
            self.broken = True
            raise Exception(f"Mechanical Failure: {self.fault_type}")

        v_from = self.speed / 60
        efficiency = max(0.1,
                         (0.5 + (0.5 * h)) - (0.2 if self.fault_active and self.fault_type == "imbalance" else 0))
        self.speed = (self.speed + (efficiency * self.speed_desired)) / 2
        v_to = self.speed / 60

        # THERMAL DYNAMICS (Updated to use instance ambient temperature)
        friction_heat = 1.5 if (self.fault_active and self.fault_type == "bearing") else 1.0
        heat_generation = (self.speed / 10) * (2 - h) * friction_heat

        # target_temp now reflects current environmental conditions
        target_temp = self.ambient_temperature + heat_generation

        thermal_rate = 0.005 + (0.01 * self.speed / 1000)
        self.temperature += (target_temp - self.temperature) * thermal_rate

        self.pressure += ((self.speed * 2) / max(h, 0.1) - self.pressure) * 0.1

        # Vibration Signature logic
        modified_A = list(self.A)
        if self.fault_active and self.fault_type == "imbalance":
            modified_A[0] *= (5.0 * (1.1 - h))

        vibration_health_modifier = 1.0 + (1.0 - h) * 5
        final_A = [amp * vibration_health_modifier for amp in modified_A]

        sample = VibrationSensorSignalSample(
            self.W, final_A, v_from, v_to, health_index=h, fault_type=self.fault_type, t=self.t
        )

        self.t += 1
        return {
            'machineID': self.name,
            'speed_desired': round(self.speed_desired, 2),
            'speed': round(self.speed + random.uniform(-5, 5), 2),
            'temperature': round(self.temperature + random.uniform(-0.1, 0.1), 2),
            'pressure': round(self.pressure + random.uniform(-20, 20), 2),
            'ambient_temperature': round(self.ambient_temperature, 2),  # Added for transparency
            'health_index': round(h, 4),
            'fault_type': self.fault_type,
            'fault_active': int(self.fault_active),
            'vibration': sample
        }
