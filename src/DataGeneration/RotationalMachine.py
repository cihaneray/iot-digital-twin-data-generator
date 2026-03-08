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
    MIN_PRESSURE: float = 70.0  # kPa — Azure-aligned lower bound

    def __init__(self, name: str, config: GlobalSimulationConfig):
        self.name = name
        self.config = config

        # Per-machine RNG — seeded deterministically, isolated from other machines
        name_hash = int(hashlib.md5(name.encode()).hexdigest(), 16)
        self._rng = random.Random(config.seed ^ (name_hash & 0xFFFF_FFFF_FFFF_FFFF))

        np_seed = (config.seed + name_hash) & 0xFFFF_FFFF
        self._np_rng = np.random.default_rng(np_seed)

        self.W = [1 / 2, 1, 2, 3, 5, 7, 12, 18]
        self.A = [1, 5, 80, 2 / 3, 8, 2, 14, 50]

        # With load_sag=15V and fault_sag=25V, the operating range is [150, 250] V.
        self._voltage_base: float = self._rng.uniform(190.0, 250.0)

        # All mutable state is owned by reset(); call it to initialise.
        self.reset()

    def reset(self):
        """
        Resets the machine to a factory-fresh state.
        Called on initialization and after a repair event.
        """
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

        # Uses the machine's own RNG, so fault assignment is reproducible per machine
        self.fault_type: str = self._assign_fault_type()

        # Scalar phase accumulator — avoids a linked-list chain of sample objects
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
        """
        Base wear-rate coefficient (per second) *before* the damage-acceleration
        factor is applied.

        The full wear ODE is:
            d(damage)/dt = base_rate * (1 + 0.5 * damage)

        Separating base_rate from the acceleration term allows the analytical
        RUL estimator (GenerateData._estimate_failure_time) to solve the ODE
        in closed form rather than stepping second-by-second.

        FIX #11: `load` (0–100 %) is now an explicit input.
            load_factor = 0.5 + 0.5 * (load / 100)
            At load_base=85 %: factor ≈ 0.925
            At load=0 %:       factor = 0.50  (idle spin, half the wear)
            At load=100 %:     factor = 1.00  (full load)
        """
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

        # Speed factor: kinetic energy proxy (v²)
        speed_factor = (speed / 1000.0) ** 2

        # Thermal stress (Arrhenius-like, kicks in above 80 °C)
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
        """
        Full per-second wear delta = base_rate × acceleration_factor.
        """
        base_rate = RotationalMachine.compute_base_wear_rate(
            stress_multiplier, speed, load, temperature, fault_active, fault_type
        )
        # Exponential acceleration: wear rate grows with accumulated damage
        acceleration_factor = 1.0 + accumulated_damage * 0.5
        return base_rate * acceleration_factor

    def _calculate_wear_delta(self) -> float:
        """Instance wrapper — passes the current machine state to the static wear model."""
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
        """
        Probabilistic fault onset below health = 0.85.
        Once active, a fault never deactivates (physical irreversibility).
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

    def _update_load(self):
        """
        At full speed (1000 RPM) load converges to config.load_base (default 85 %).
        At zero speed load decays to zero.  The lag (rate = 0.1) is slower than
        speed dynamics so that sudden speed changes produce a realistic load
        transient rather than an instantaneous jump.
        """
        target_load = (self.speed / 1000.0) * self.config.load_base
        self.load += (target_load - self.load) * 0.1
        self.load = max(0.0, min(100.0, self.load))

    def _update_temperature(self):
        """
        Thermal model.

        Calibration targets at full speed (1000 RPM):
            Healthy, no fault          h=1.0: ~75 °C   (well within operating range)
            Degraded, no fault         h=0.0: ~102 °C  (elevated but survivable)
            End-of-life, bearing fault h=0.0: ~122 °C  → Over-Temperature Failure ✓

        Coefficient 53.3 was chosen so that OTF (120 °C) is only reachable for
        machines that are both severely degraded AND have active bearing friction.
        The previous value of 45.0 made OTF physically unreachable (max ~106 °C).
        """
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
        """
        Pressure is load-dependent (speed proxy) only — not health-dependent —
        to avoid feature leakage that would let ML models "cheat" via pressure.

        Sensor jitter (±1.5 kPa) is applied at read time in next_state, consistent
        with all other sensors.  Previously ±2 kPa noise was added inside the lag
        update, which compounded into the filter state and caused ~2 kPa of
        unbounded drift.
        """
        target = self.config.pressure_base * (0.8 + 0.2 * (self.speed / 1000.0))
        self.pressure += (target - self.pressure) * 0.2
        self.pressure = max(self.MIN_PRESSURE, self.pressure)

    def _update_voltage(self):
        """
        Per-machine base voltage (set once at birth): U[190, 250] V
        Load sag at full speed:  −15 V  → min with no fault = 175 V
        Fault sag (dead bearing): −25 V  → min overall     = 150 V  ✓

        Previously: fixed base 230 V, load sag 5 V, fault sag 10 V → range [215, 232],
        which was far narrower than the 150–250 V Azure target.
        """
        load_sag = 15.0 * (self.speed / 1000.0)

        fault_sag = 0.0
        if self.fault_active and self.fault_type == "bearing":
            fault_sag = 25.0 * (1.0 - self.health_index)

        noise = self._rng.uniform(-2.0, 2.0)
        self.voltage = self._voltage_base - load_sag - fault_sag + noise

    def next_state(self, rul_seconds: float | None = None) -> dict:
        """
        Advance the machine by one second and return the observable state.

        Parameters
        ----------
        rul_seconds : float | None
            Pre-computed remaining useful life in seconds (supplied by GenerateData).
            Written directly to the output column when provided; None omits it.

        Returns
        -------
        dict
            Scalar sensor readings plus ``vibration_pcm`` (numpy int16 array).
            The caller (GenerateData) pops ``vibration_pcm`` and handles storage.

        Raises
        ------
        Exception
            "Mechanical Failure: <fault_type>" when health reaches zero.
            "Over-Temperature Failure" when temperature reaches MAX_TEMPERATURE.
        """
        if self._over_temp:
            raise Exception("Over-Temperature Failure")
        if self.broken:
            raise Exception(f"Mechanical Failure: {self.fault_type}")

        # 1. Fault onset (probabilistic)
        self._update_fault_active()

        # 2. Damage accumulation
        self.accumulated_damage += self._calculate_wear_delta()
        h = self.health_index

        # 3. Terminal failure checks
        if h <= 0.0:
            self.broken = True
            raise Exception(f"Mechanical Failure: {self.fault_type}")

        if self.temperature >= self.MAX_TEMPERATURE:
            self._over_temp = True
            raise Exception("Over-Temperature Failure")

        # 4. Speed dynamics — first-order lag toward target
        efficiency = max(0.1, 0.5 + 0.5 * h)
        inertia = 0.05
        if self.speed < self.speed_desired:
            self.speed += (self.speed_desired - self.speed) * inertia * efficiency
        else:
            self.speed += (self.speed_desired - self.speed) * inertia * 1.5

        v_from = self.speed / 60.0
        v_to = self.speed / 60.0

        # 5. Subsystem updates
        self._update_load()
        self._update_temperature()
        self._update_pressure()
        self._update_voltage()

        # 6. Vibration signal generation
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

        # 8. Assemble output — sensor noise applied at read time only
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
