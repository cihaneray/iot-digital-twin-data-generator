from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class GlobalSimulationConfig:
    seed: int
    date_from: datetime
    date_to: datetime
    telemetry_batch_frequency: timedelta
    telemetry_interval: timedelta
    machine_count: int
    active_machines_per_batch: int
    cycle_length_min: int
    cycle_length_max: int

    # Stress and Degradation Controls
    stress_multiplier: float = 0.25  # Lowered for realistic wear
    failure_threshold: float = 5.0  # Lowered to force failures in a timeframe

    # Environmental Cycle Parameters
    ambient_temp_mean: float = 22.0
    ambient_temp_amplitude: float = 8.0

    # Fault Probabilities
    imbalance_probability: float = 0.1
    bearing_fault_probability: float = 0.1

    def __post_init__(self):
        assert self.date_from < self.date_to, "Date from must be before date to."
