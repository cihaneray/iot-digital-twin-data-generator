from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class GlobalSimulationConfig:
    seed: int
    date_from: datetime
    date_to: datetime
    telemetry_batch_frequency: timedelta
    machine_count: int
    active_machines_per_batch: int
    cycle_length_min: int
    cycle_length_max: int

    # Stress and Degradation Controls
    stress_multiplier: float = 0.25
    failure_threshold: float = 5.0

    # Environmental Cycle Parameters
    ambient_temp_mean: float = 22.0
    ambient_temp_amplitude: float = 8.0

    # Fault Probabilities
    imbalance_probability: float = 0.1
    bearing_fault_probability: float = 0.1

    # Fault onset: higher = more abrupt transition (logistic curve steepness)
    fault_onset_sharpness: float = 20.0

    # Health threshold where a designated fault is allowed to become active.
    fault_activation_health_threshold: float = 0.85

    # Health threshold where fault vibration signature reaches full strength.
    # Between activation threshold and this value, amplitudes are ramped gradually.
    vibration_full_signature_health: float = 0.80

    # Fault vibration gain controls. Fault signatures scale with speed, load, and health.
    fault_vibration_speed_exponent: float = 1.5
    fault_vibration_min_speed_ratio: float = 0.02
    fault_vibration_load_influence: float = 0.5
    fault_vibration_health_exponent: float = 1.0

    # Harmonic spectrum controls for operating-point-dependent baseline vibration.
    harmonic_rolloff_exponent: float = 1.6
    harmonic_resonance_strength: float = 0.25
    base_signal_rms_nominal: float = 28.0
    signal_rms_budget: float = 55.0

    # Bearing shock controls. Shocks are gated by speed and scaled by load.
    bearing_shock_min_speed_ratio: float = 0.35
    bearing_shock_transition_ratio: float = 0.2
    bearing_shock_load_influence: float = 0.35
    bearing_shock_base_magnitude: float = 40.0

    # Physical Limits and Nominal Values
    max_temperature: float = 120.0
    min_pressure: float = 70.0
    nominal_speed: float = 1000.0

    # Nominal operational load (%) at full speed.
    # Drives the load_factor in wear calculation: 0.5 + 0.5 * (load / 100).
    # At load_base=85%: factor ≈ 0.925.  At 0%: a factor = 0.5. At 100%: a factor = 1.0.
    load_base: float = 85.0

    # Pressure target (kPa). Azure PDM typical range: 75–115 kPa.
    pressure_base: float = 100.0

    # Voltage nominal (V). Per-machine base is drawn from U[190, 250] at birth;
    # this value is the center of that distribution.
    # Azure PDM typical range: 150–250 V.
    voltage_base: float = 230.0

    # Vibration sample rate (Hz). 1024 captures harmonics up to ~500 Hz.
    vibration_sample_rate: int = 1024

    # "pyarrow" (default)
    # Switch to "fastparquet" if your environment requires it.
    parquet_engine: str = "pyarrow"

    # Error log flush interval (in batches). Keeps RAM usage bounded.
    error_flush_interval: int = 50

    def __post_init__(self):
        assert self.date_from < self.date_to, "date_from must be before date_to."
        assert self.cycle_length_min <= self.cycle_length_max, (
            "cycle_length_min must be <= cycle_length_max."
        )
        assert self.cycle_length_max <= 60, "cycle_length_max must be <= 60 minutes."
        assert self.active_machines_per_batch <= self.machine_count, (
            "active_machines_per_batch cannot exceed machine_count."
        )
        assert 0.0 <= self.imbalance_probability <= 1.0, (
            "imbalance_probability must be in [0, 1]."
        )
        assert 0.0 <= self.bearing_fault_probability <= 1.0, (
            "bearing_fault_probability must be in [0, 1]."
        )
        assert (self.imbalance_probability + self.bearing_fault_probability) <= 1.0, (
            "imbalance_probability + bearing_fault_probability must not exceed 1.0."
        )
        assert 0.0 <= self.vibration_full_signature_health < self.fault_activation_health_threshold <= 1.0, (
            "vibration_full_signature_health must be < fault_activation_health_threshold, both in [0, 1]."
        )
        assert self.fault_vibration_speed_exponent > 0.0, (
            "fault_vibration_speed_exponent must be > 0."
        )
        assert 0.0 <= self.fault_vibration_min_speed_ratio <= 1.0, (
            "fault_vibration_min_speed_ratio must be in [0, 1]."
        )
        assert 0.0 <= self.fault_vibration_load_influence <= 1.0, (
            "fault_vibration_load_influence must be in [0, 1]."
        )
        assert self.fault_vibration_health_exponent > 0.0, (
            "fault_vibration_health_exponent must be > 0."
        )
        assert self.harmonic_rolloff_exponent > 0.0, (
            "harmonic_rolloff_exponent must be > 0."
        )
        assert 0.0 <= self.harmonic_resonance_strength <= 1.0, (
            "harmonic_resonance_strength must be in [0, 1]."
        )
        assert self.base_signal_rms_nominal > 0.0, (
            "base_signal_rms_nominal must be > 0."
        )
        assert self.signal_rms_budget > 0.0, (
            "signal_rms_budget must be > 0."
        )
        assert 0.0 <= self.bearing_shock_min_speed_ratio <= 1.5, (
            "bearing_shock_min_speed_ratio must be in [0, 1.5]."
        )
        assert self.bearing_shock_transition_ratio > 0.0, (
            "bearing_shock_transition_ratio must be > 0."
        )
        assert 0.0 <= self.bearing_shock_load_influence <= 1.0, (
            "bearing_shock_load_influence must be in [0, 1]."
        )
        assert self.bearing_shock_base_magnitude > 0.0, (
            "bearing_shock_base_magnitude must be > 0."
        )
        assert 0.0 < self.load_base <= 100.0, "load_base must be in (0, 100]."
        assert self.parquet_engine in ("pyarrow", "fastparquet"), (
            "parquet_engine must be 'pyarrow' or 'fastparquet'."
        )
