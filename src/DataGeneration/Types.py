from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class SimulationParameters:
    seed: int
    date_from: datetime
    date_to: datetime
    telemetry_batch_frequency: timedelta
    telemetry_interval: timedelta
    machine_count: int
    active_machines_per_batch: int
    cycle_length_min: int
    cycle_length_max: int

    def __post_init__(self):
        assert self.date_from < self.date_to, "Date from must be before date to."


@dataclass
class SensorParameters:
    t_max: int = 10000
    d: float = 0.05
    a: float = -0.3
    b: float = 0.2
    th: float = 0.45
