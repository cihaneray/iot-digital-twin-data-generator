import math
import random
import hashlib

import numpy as np

from .Types import GlobalSimulationConfig
from .VibrationSensorSignalSample import VibrationSensorSignalSample


class RotationalMachine:
    """
    Physics-based simulation of a rotational machine with degradation modeling.
    Aligned to standard Predictive Maintenance datasets (e.g., Azure PDM).
    """

    MAX_TEMPERATURE: float = 120.0

    # Azure-aligned range is ~75-115 kPa
    MIN_PRESSURE: float = 70.0

    def __init__(self, name: str, config: GlobalSimulationConfig):
        self.name = name
        self.config = config

        # Per-machine RNG — seeded deterministically but isolated from other machines
        name_hash = int(hashlib.md5(name.encode()).hexdigest(), 16)
        self._rng = random.Random(config.seed ^ (name_hash & 0xFFFF_FFFF_FFFF_FFFF))

        np_seed = (config.seed + name_hash) & 0xFFFF_FFFF
        self._np_rng = np.random.default_rng(np_seed)

        # Harmonic components (shared across all instances)
        self.W = [1 / 2, 1, 2, 3, 5, 7, 12, 18]
        self.A = [1, 5, 80, 2 / 3, 8, 2, 14, 50]

        self.reset()

    def reset(self):
        """
        Resets the machine to a factory-fresh state.
        Called on initialization and after a repair event.
        """
        self.t: int = 0
        self.speed: float = 0.0
        self.speed_desired: float = 0.0
        self.ambient_temperature: float = self.config.ambient_temp_mean
        self.temperature: float = self.ambient_temperature
        self.pressure: float = self.config.pressure_base
        self.voltage: float = self.config.voltage_base
        self.accumulated_damage: float = 0.0
        self.broken: bool = False
        self.fault_active: bool = False
        self._over_temp: bool = False

        # Uses the machine's own RNG, so fault assignment is reproducible per machine
        self.fault_type: str = self._assign_fault_type()

        # Scalar phase accumulator — avoids holding a chain of sample objects
        self._last_cumsum: float = 0.0

    def _assign_fault_type(self) -> str:
        """
        Assigns a fault type at machine birth using weighted random selection.
        """
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
    def compute_wear_step(
            stress_multiplier: float,
            speed: float,
            temperature: float,
            fault_active: bool,
            fault_type: str,
            accumulated_damage: float
    ) -> float:
        """
        Pure function to calculate wear delta.
        CRITICAL: This logic is shared by the live simulation and the RUL Estimator
        in GenerateData.py. Any changes here must be physically sound for both.
        """
        # Wear accumulates as long as the machine is moving (speed > 0)
        if speed <= 0.1:
            return 0.0

        base_wear = 0.001 * stress_multiplier

        fault_multiplier = 1.0
        if fault_active:
            if fault_type == "bearing":
                fault_multiplier = 1.2
            elif fault_type == "imbalance":
                fault_multiplier = 1.1

        # Exponential wear model: Wear rate increases as damage increases
        acceleration_factor = 1.0 + (accumulated_damage * 0.5)

        # Thermal stress factor (starts kicking in above 80C)
        temp_factor = 1.0
        if temperature > 80.0:
            temp_factor = math.exp((temperature - 80.0) / 20.0)

        # Speed factor: wear is proportional to kinetic energy (v^2)
        speed_factor = (speed / 1000.0) ** 2

        return base_wear * speed_factor * temp_factor * fault_multiplier * acceleration_factor

    def _calculate_wear_delta(self) -> float:
        """
        Instance wrapper for the static wear logic.
        """
        return self.compute_wear_step(
            self.config.stress_multiplier,
            self.speed,
            self.temperature,
            self.fault_active,
            self.fault_type,
            self.accumulated_damage
        )

    def _update_fault_active(self):
        """
        Probabilistic fault onset.
        """
        if self.fault_active or self.fault_type == "normal":
            return

        h = self.health_index
        if h >= 0.85:
            return

        deficit = 0.85 - h
        probability = 1.0 / (1.0 + math.exp(-self.config.fault_onset_sharpness * deficit))

        if self._rng.random() < probability:
            self.fault_active = True

    def _update_temperature(self):
        """
        Thermal model calibrated to avoid premature 'Over-Temperature' aborts.

        New Calibration:
        - Healthy @ Full Speed: ~65°C (was ~77°C)
        - Dead Bearing @ Full Speed: ~112°C (was ~187°C)
        - Max Limit: 120°C
        """
        friction_heat = 1.0
        if self.fault_active:
            # Reduced multipliers to prevent unrealistic thermal runaway
            if self.fault_type == "bearing":
                friction_heat = 1.25  # Was 1.5
            elif self.fault_type == "imbalance":
                friction_heat = 1.1  # Was 1.2

        # Efficiency drops as health drops.
        # Capped at 1.5x penalty (was 2.0x) to prevent explosion at h=0
        health_penalty = 1.0 + (0.5 * (1.0 - self.health_index))

        heat_generation = (self.speed / 1000.0) * 53.3 * health_penalty * friction_heat

        target_temp = self.ambient_temperature + heat_generation

        # Thermal inertia
        thermal_rate = 0.005 + (0.01 * self.speed / 1000.0)
        self.temperature += (target_temp - self.temperature) * thermal_rate
        self.temperature = min(self.temperature, self.MAX_TEMPERATURE)

    def _update_pressure(self):
        """
        Azure Alignment: Pressure should be ~75-115 kPa.
        Pressure is primarily load-dependent (speed), NOT health-dependent,
        to prevent 'feature leakage' (ML models cheating via pressure drops).
        """
        # Nominal pressure at full speed is config base (e.g., 100)
        # Pressure drops at low speed.
        target = self.config.pressure_base * (0.8 + 0.2 * (self.speed / 1000.0))

        # Add process noise
        noise = self._rng.uniform(-2.0, 2.0)

        # Pressure lag
        self.pressure += (target - self.pressure) * 0.2 + noise
        self.pressure = max(self.MIN_PRESSURE, self.pressure)

    def _update_voltage(self):
        """
        Azure Alignment: Voltage ~150-250V.
        Voltage sags under load and when faults are active (increased resistance/drag).
        """
        base_v = self.config.voltage_base

        # Load sag: Voltage drops slightly as speed/load increases
        load_sag = 5.0 * (self.speed / 1000.0)

        # Fault sag: Significant drop if bearings are seizing
        fault_sag = 0.0
        if self.fault_active and self.fault_type == "bearing":
            fault_sag = 10.0 * (1.0 - self.health_index)

        # Electrical noise
        noise = self._rng.uniform(-2.0, 2.0)

        self.voltage = base_v - load_sag - fault_sag + noise

    def next_state(self, rul_seconds: float | None = None) -> dict:
        """
        Advance the machine one second.
        Returns the Azure-compatible telemetry dict.

        Parameters
        ----------
        rul_seconds : float | None
            Pre-computed remaining useful life in seconds (from GenerateData).
            When provided, the value is written directly to the ``rul_seconds``
            output column. Pass None to omit the column (default).
        """
        if self._over_temp:
            raise Exception("Over-Temperature Failure")
        if self.broken:
            raise Exception(f"Mechanical Failure: {self.fault_type}")

        # 1. Fault onset
        self._update_fault_active()

        # 2. Accumulate damage (using centralized logic)
        self.accumulated_damage += self._calculate_wear_delta()
        h = self.health_index

        if h <= 0.0:
            self.broken = True
            raise Exception(f"Mechanical Failure: {self.fault_type}")

        if self.temperature >= self.MAX_TEMPERATURE:
            self._over_temp = True
            raise Exception("Over-Temperature Failure")

        # 3. Speed dynamics (Corrected: First-order lag)
        # Efficiency determines how fast it can speed up
        efficiency = max(0.1, (0.5 + 0.5 * h))
        inertia = 0.05  # Time constant

        if self.speed < self.speed_desired:
            # Accelerate
            self.speed += (self.speed_desired - self.speed) * inertia * efficiency
        else:
            # Decelerate (friction helps stop)
            self.speed += (self.speed_desired - self.speed) * inertia * 1.5

        v_from = self.speed / 60.0
        v_to = self.speed / 60.0  # Simplified for 1s step

        # 4. Updates
        self._update_temperature()
        self._update_pressure()
        self._update_voltage()

        # 5. Vibration Generation
        # Logic modified to handle signal amplitude scaling for Azure compatibility
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
            rng=self._np_rng
        )

        pcm = sample.pcm()
        self._last_cumsum = sample.last_cumsum
        self.t += 1

        # AZURE ALIGNMENT: Calculate scalar vibration intensity (RMS)
        # This matches the "vibration" column in Azure (typical range 20-50)
        # We normalize int16 to float, calculate RMS, and scale to fit range.
        float_signal = pcm.astype(np.float32) / 32767.0
        rms = np.sqrt(np.mean(float_signal ** 2))
        # Scale factor 100 arbitrary to map low-amplitude signals to ~20-50 range
        vibration_intensity = 20.0 + (rms * 100.0)

        reported_speed = max(0.0, self.speed + self._rng.uniform(-5, 5))

        state = {
            "machineID": self.name,
            "speed_desired": round(self.speed_desired, 2),
            "speed": round(reported_speed, 2),
            "temperature": round(self.temperature + self._rng.uniform(-0.1, 0.1), 2),
            "pressure": round(self.pressure, 2),
            "ambient_temperature": round(self.ambient_temperature, 2),
            "voltage": round(self.voltage, 2),
            "vibration_intensity": round(vibration_intensity, 2),
            "health_index": round(h, 4),
            "fault_type": self.fault_type,
            "fault_active": int(self.fault_active),
            "vibration_pcm": pcm,
        }

        if rul_seconds is not None:
            state["rul_seconds"] = max(0.0, rul_seconds)
        else:
            state["rul_seconds"] = None

        return state
