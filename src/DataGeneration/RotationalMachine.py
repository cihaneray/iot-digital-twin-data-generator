import math
import random
import hashlib

import numpy as np

from .Types import GlobalSimulationConfig
from .VibrationSensorSignalSample import VibrationSensorSignalSample


class RotationalMachine:
    MAX_TEMPERATURE: float = 120.0
    MIN_PRESSURE: float = 70.0

    def __init__(self, name: str, config: GlobalSimulationConfig):
        self._last_cumsum = None
        self.broken = None
        self._over_temp = None
        self.speed_desired: float = 0.0
        self.name = name
        self.config = config

        name_hash = int(hashlib.md5(name.encode()).hexdigest(), 16)
        self._rng = random.Random(config.seed ^ (name_hash & 0xFFFF_FFFF_FFFF_FFFF))

        np_seed = (config.seed + name_hash) & 0xFFFF_FFFF
        self._np_rng = np.random.default_rng(np_seed)

        self.W = [1 / 2, 1, 2, 3, 5, 7, 12, 18]
        self.A = [1, 5, 80, 2 / 3, 8, 2, 14, 50]

        self._voltage_base: float = self._rng.uniform(190.0, 250.0)

        self.__reset__()

    def __reset__(self):
        self.t: int = 0
        self.speed: float = 0.0
        self.speed_desired: float = 0.0
        self.load: float = 0.0
        self.ambient_temperature: float = self.config.ambient_temp_mean
        self.temperature: float = self.ambient_temperature
        self.pressure: float = self.config.pressure_base
        self.voltage: float = self._voltage_base
        self.accumulated_damage: float = 0.0
        self.broken: bool = False
        self.fault_active: bool = False
        self._over_temp: bool = False

        self.fault_type: str = self._assign_fault_type()

        self._last_cumsum: float = 0.0

    def _assign_fault_type(self) -> str:
        normal_probability = (
                1.0
                - self.config.imbalance_probability
                - self.config.bearing_fault_probability
        )
        fault_types = ["imbalance", "bearing", "normal"]
        weights = [
            self.config.imbalance_probability,
            self.config.bearing_fault_probability,
            normal_probability,
        ]
        return self._rng.choices(fault_types, weights=weights, k=1)[0]

    @property
    def health_index(self) -> float:
        return max(0.0, 1.0 - (self.accumulated_damage / self.config.failure_threshold))

    def set_speed(self, speed: float):
        self.speed_desired = speed

    @staticmethod
    def compute_base_wear_rate(
            stress_multiplier: float,
            speed: float,
            load: float,
            temperature: float,
            fault_active: bool,
            fault_type: str,
    ) -> float:
        if speed <= 0.1:
            return 0.0

        base_wear = 0.001 * stress_multiplier

        fault_multiplier = 1.0
        if fault_active:
            if fault_type == "bearing":
                fault_multiplier = 1.2
            elif fault_type == "imbalance":
                fault_multiplier = 1.1

        load_factor = 0.5 + 0.5 * (load / 100.0)

        speed_factor = (speed / 1000.0) ** 2

        temp_factor = (
            math.exp((temperature - 80.0) / 20.0)
            if temperature > 80.0
            else 1.0
        )

        return base_wear * speed_factor * load_factor * temp_factor * fault_multiplier

    @staticmethod
    def compute_wear_step(
            stress_multiplier: float,
            speed: float,
            load: float,
            temperature: float,
            fault_active: bool,
            fault_type: str,
            accumulated_damage: float,
    ) -> float:
        base_rate = RotationalMachine.compute_base_wear_rate(
            stress_multiplier, speed, load, temperature, fault_active, fault_type
        )
        acceleration_factor = 1.0 + accumulated_damage * 0.5
        return base_rate * acceleration_factor

    def _calculate_wear_delta(self) -> float:
        return self.compute_wear_step(
            self.config.stress_multiplier,
            self.speed,
            self.load,
            self.temperature,
            self.fault_active,
            self.fault_type,
            self.accumulated_damage,
        )

    def _update_fault_active(self):
        if self.fault_active or self.fault_type == "normal":
            return

        h = self.health_index
        if h >= 0.85:
            return

        deficit = 0.85 - h
        probability = 1.0 / (1.0 + math.exp(-self.config.fault_onset_sharpness * deficit))

        if self._rng.random() < probability:
            self.fault_active = True

    def _update_load(self):
        target_load = (self.speed / 1000.0) * self.config.load_base
        self.load += (target_load - self.load) * 0.1
        self.load = max(0.0, min(100.0, self.load))

    def _update_temperature(self):
        friction_heat = 1.0
        if self.fault_active:
            if self.fault_type == "bearing":
                friction_heat = 1.25
            elif self.fault_type == "imbalance":
                friction_heat = 1.1

        health_penalty = 1.0 + 0.5 * (1.0 - self.health_index)
        heat_generation = (self.speed / 1000.0) * 53.3 * health_penalty * friction_heat

        target_temp = self.ambient_temperature + heat_generation
        thermal_rate = 0.005 + 0.01 * (self.speed / 1000.0)
        self.temperature += (target_temp - self.temperature) * thermal_rate
        self.temperature = min(self.temperature, self.MAX_TEMPERATURE)

    def _update_pressure(self):
        target = self.config.pressure_base * (0.8 + 0.2 * (self.speed / 1000.0))
        self.pressure += (target - self.pressure) * 0.2
        self.pressure = max(self.MIN_PRESSURE, self.pressure)

    def _update_voltage(self):
        load_sag = 15.0 * (self.speed / 1000.0)

        fault_sag = 0.0
        if self.fault_active and self.fault_type == "bearing":
            fault_sag = 25.0 * (1.0 - self.health_index)

        noise = self._rng.uniform(-2.0, 2.0)
        self.voltage = self._voltage_base - load_sag - fault_sag + noise

    def next_state(self, rul_seconds: float | None = None) -> dict:
        if self._over_temp:
            raise Exception("Over-Temperature Failure")
        if self.broken:
            raise Exception(f"Mechanical Failure: {self.fault_type}")

        self._update_fault_active()

        self.accumulated_damage += self._calculate_wear_delta()
        h = self.health_index

        if h <= 0.0:
            self.broken = True
            raise Exception(f"Mechanical Failure: {self.fault_type}")

        if self.temperature >= self.MAX_TEMPERATURE:
            self._over_temp = True
            raise Exception("Over-Temperature Failure")

        efficiency = max(0.1, 0.5 + 0.5 * h)
        inertia = 0.05
        if self.speed < self.speed_desired:
            self.speed += (self.speed_desired - self.speed) * inertia * efficiency
        else:
            self.speed += (self.speed_desired - self.speed) * inertia * 1.5

        v_from = self.speed / 60.0
        v_to = self.speed / 60.0

        self._update_load()
        self._update_temperature()
        self._update_pressure()
        self._update_voltage()

        modified_A = list(self.A)
        if self.fault_active and self.fault_type == "imbalance":
            modifier = 4.0 * (0.85 - h) / 0.85
            modified_A[0] *= max(1.0, 1.0 + modifier)

        sample = VibrationSensorSignalSample(
            w=self.W,
            a=modified_A,
            fundamental_from=v_from,
            fundamental_to=v_to,
            health_index=h,
            fault_type=self.fault_type,
            t=self.t,
            prev_cumsum=self._last_cumsum,
            sample_rate=self.config.vibration_sample_rate,
            rng=self._np_rng,
        )

        pcm = sample.pcm()
        self._last_cumsum = sample.last_cumsum
        self.t += 1

        float_signal = pcm.astype(np.float32) / 32767.0
        rms = float(np.sqrt(np.mean(float_signal ** 2)))
        vibration_intensity = 20.0 + 30.0 * math.tanh(rms * 3.5)

        reported_speed = max(0.0, self.speed + self._rng.uniform(-5.0, 5.0))
        reported_load = max(0.0, self.load + self._rng.uniform(-2.0, 2.0))
        reported_press = max(self.MIN_PRESSURE,
                             self.pressure + self._rng.uniform(-1.5, 1.5))

        state = {"machineID": self.name, "speed_desired": round(self.speed_desired, 2),
                 "speed": round(reported_speed, 2), "load": round(reported_load, 2),
                 "temperature": round(self.temperature + self._rng.uniform(-0.1, 0.1), 2),
                 "pressure": round(reported_press, 2), "ambient_temperature": round(self.ambient_temperature, 2),
                 "voltage": round(self.voltage, 2), "vibration_intensity": round(vibration_intensity, 2),
                 "health_index": round(h, 4), "fault_type": self.fault_type, "fault_active": int(self.fault_active),
                 "vibration_pcm": pcm, "rul_seconds": max(0.0, rul_seconds) if rul_seconds is not None else None}

        return state
