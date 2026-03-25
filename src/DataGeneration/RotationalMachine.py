import math
import random
import hashlib

import numpy as np

from .Types import GlobalSimulationConfig
from .VibrationSensorSignalSample import VibrationSensorSignalSample


class RotationalMachine:
    def __init__(self, name: str, config: GlobalSimulationConfig):
        self._last_cumsum = None # type: ignore
        self.broken = None # type: ignore
        self._over_temp = None # type: ignore
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
        self._over_temp: bool = False

        self.designated_fault: str = self._assign_fault_type()
        self.active_faults: set[str] = set()
        self._imbalance_cascade_rolled: bool = False

        self._last_cumsum: float = 0.0

    @property
    def fault_active(self) -> bool:
        return len(self.active_faults) > 0

    @property
    def fault_type(self) -> str:
        if not self.active_faults:
            return "normal"
        return "+".join(sorted(self.active_faults))

    def _assign_fault_type(self) -> str:
        # Ensure probabilities sum to 1 for exclusive fault assignment
        total_fault_prob = self.config.imbalance_probability + self.config.bearing_fault_probability
        if total_fault_prob > 1.0:
            # Normalize if sum exceeds 1, or handle as an error/warning
            # For now, we'll just cap it and distribute the remainder to normal
            imbalance_prob = self.config.imbalance_probability / total_fault_prob
            bearing_prob = self.config.bearing_fault_probability / total_fault_prob
            normal_probability = 0.0
        else:
            imbalance_prob = self.config.imbalance_probability
            bearing_prob = self.config.bearing_fault_probability
            normal_probability = 1.0 - total_fault_prob

        fault_types = ["imbalance", "bearing", "normal"]
        weights = [
            imbalance_prob,
            bearing_prob,
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
            active_faults: set[str],
            nominal_speed: float = 1000.0,
    ) -> float:
        if speed <= 0.1:
            return 0.0

        base_wear = 0.001 * stress_multiplier

        fault_multiplier = 1.0
        if "bearing" in active_faults:
            fault_multiplier *= 1.2
        if "imbalance" in active_faults:
            fault_multiplier *= 1.1

        load_factor = 0.5 + 0.5 * (load / 100.0)
        speed_factor = (speed / nominal_speed) ** 2

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
            active_faults: set[str],
            accumulated_damage: float,
            nominal_speed: float = 1000.0,
    ) -> float:
        base_rate = RotationalMachine.compute_base_wear_rate(
            stress_multiplier, speed, load, temperature, active_faults, nominal_speed
        )
        acceleration_factor = 1.0 + accumulated_damage * 0.5
        return base_rate * acceleration_factor

    def _calculate_wear_delta(self) -> float:
        return self.compute_wear_step(
            self.config.stress_multiplier,
            self.speed,
            self.load,
            self.temperature,
            self.active_faults,
            self.accumulated_damage,
            self.config.nominal_speed,
        )

    def _update_fault_active(self):
        h = self.health_index
        # Activate primary designated fault
        if self.designated_fault != "normal" and self.designated_fault not in self.active_faults:
            if h < self.config.fault_activation_health_threshold:
                deficit = self.config.fault_activation_health_threshold - h
                probability = 1.0 / (1.0 + math.exp(-self.config.fault_onset_sharpness * deficit))
                if self._rng.random() < probability:
                    self.active_faults.add(self.designated_fault)

        # Cascade faults: Severe bearing wear has a 30% chance to induce an imbalance
        if "bearing" in self.active_faults and "imbalance" not in self.active_faults:
            if h < 0.5 and not self._imbalance_cascade_rolled:
                self._imbalance_cascade_rolled = True
                if self._rng.random() < 0.30:
                    self.active_faults.add("imbalance")

    def _update_temperature(self):
        friction_heat = 1.0
        if "bearing" in self.active_faults:
            friction_heat *= 1.25
        if "imbalance" in self.active_faults:
            friction_heat *= 1.1

        # High load increases temperature slightly
        load_penalty = 1.0 + 0.2 * (self.load / 100.0)

        health_penalty = 1.0 + 0.5 * (1.0 - self.health_index)
        heat_generation = (self.speed / self.config.nominal_speed) * 50.0 * health_penalty * friction_heat * load_penalty

        target_temp = self.ambient_temperature + heat_generation
        thermal_rate = 0.005 + 0.01 * (self.speed / self.config.nominal_speed)
        self.temperature += (target_temp - self.temperature) * thermal_rate
        self.temperature = min(self.temperature, self.config.max_temperature)

    def _update_load(self):
        base_target_load = (self.speed / self.config.nominal_speed) * self.config.load_base
        
        # High temperature reduces viscosity, increasing friction and thus motor load
        temp_penalty = 1.0
        if self.temperature > 80.0:
            temp_penalty += 0.05 * ((self.temperature - 80.0) / 10.0)
            
        health_penalty = 1.0 + 0.2 * (1.0 - self.health_index)
        target_load = base_target_load * temp_penalty * health_penalty

        self.load += (target_load - self.load) * 0.1
        self.load = max(0.0, min(100.0, self.load))
        
    def _update_pressure(self):
        target = self.config.pressure_base * (0.8 + 0.2 * (self.speed / self.config.nominal_speed))
        self.pressure += (target - self.pressure) * 0.2
        self.pressure = max(self.config.min_pressure, self.pressure)

    def _update_voltage(self):
        # Load directly affects voltage sag
        load_sag = 15.0 * (self.load / self.config.load_base) if self.config.load_base else 0.0

        fault_sag = 0.0
        if "bearing" in self.active_faults:
             fault_sag += 25.0 * (1.0 - self.health_index)
        if "imbalance" in self.active_faults:
             fault_sag += 10.0 * (1.0 - self.health_index)

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

        if self.temperature >= self.config.max_temperature:
            self._over_temp = True
            raise Exception("Over-Temperature Failure")

        efficiency = max(0.1, 0.5 + 0.5 * h)
        inertia = 0.05
        speed_before_update = self.speed
        if self.speed < self.speed_desired:
            self.speed += (self.speed_desired - self.speed) * inertia * efficiency
        else:
            self.speed += (self.speed_desired - self.speed) * inertia * 1.5

        v_from = max(0.0, speed_before_update / 60.0)
        v_to = max(0.0, self.speed / 60.0)

        self._update_load()
        self._update_temperature()
        self._update_pressure()
        self._update_voltage()

        # Dynamic Vibration Scaling (A proportional to speed squared)
        speed_ratio = self.speed / self.config.nominal_speed
        amplitude_scaling = speed_ratio ** 2
        load_ratio = (self.load / self.config.load_base) if self.config.load_base else 0.0
        
        modified_A = [a * amplitude_scaling for a in self.A]
        if "imbalance" in self.active_faults:
            modifier = 4.0 * (0.85 - h) / 0.85
            modified_A[0] *= max(1.0, 1.0 + modifier)

        sample = VibrationSensorSignalSample(
            w=self.W,
            a=modified_A,
            fundamental_from=v_from,
            fundamental_to=v_to,
            health_index=h,
            active_faults=frozenset(self.active_faults),
            t=self.t,
            prev_cumsum=self._last_cumsum,
            sample_rate=self.config.vibration_sample_rate,
            speed_ratio=speed_ratio,
            load_ratio=load_ratio,
            fault_activation_health_threshold=self.config.fault_activation_health_threshold,
            vibration_full_signature_health=self.config.vibration_full_signature_health,
            fault_vibration_speed_exponent=self.config.fault_vibration_speed_exponent,
            fault_vibration_min_speed_ratio=self.config.fault_vibration_min_speed_ratio,
            fault_vibration_load_influence=self.config.fault_vibration_load_influence,
            fault_vibration_health_exponent=self.config.fault_vibration_health_exponent,
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
        reported_press = max(self.config.min_pressure,
                             self.pressure + self._rng.uniform(-1.5, 1.5))

        state = {"machineID": self.name, "speed_desired": round(self.speed_desired, 2),
                 "speed": round(reported_speed, 2), "load": round(reported_load, 2),
                 "temperature": round(self.temperature + self._rng.uniform(-0.1, 0.1), 2),
                 "pressure": round(reported_press, 2), "ambient_temperature": round(self.ambient_temperature, 2),
                 "voltage": round(self.voltage, 2), "vibration_intensity": round(vibration_intensity, 2),
                 "health_index": round(h, 4), "fault_type": self.fault_type, "fault_active": int(self.fault_active),
                 "vibration_pcm": pcm, "rul_seconds": max(0.0, rul_seconds) if rul_seconds is not None else None}

        return state
