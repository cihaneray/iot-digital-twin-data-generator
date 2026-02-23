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
    stress_multiplier: float = 1.0  # Scales the speed/temp wear calculation
    failure_threshold: float = 100.0  # Damage level at which the machine fails

    def __post_init__(self):
        assert self.date_from < self.date_to, "Date from must be before date to."
