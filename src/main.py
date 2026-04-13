from datetime import timedelta
from dateutil import parser

from DataGeneration import GlobalSimulationConfig, GenerateData

config = GlobalSimulationConfig(
    seed=42,
    date_from=parser.parse("Jan 01 2026 12:00AM"),
    date_to=parser.parse("Feb 21 2026 12:00AM"),
    telemetry_batch_frequency=timedelta(hours=1),
    machine_count=10,
    active_machines_per_batch=2,
    cycle_length_min=1,
    cycle_length_max=5,

    # Degradation
    stress_multiplier=0.25,
    failure_threshold=5.0,

    # Environment
    ambient_temp_mean=22.0,
    ambient_temp_amplitude=8.0,

    # Fault probabilities (must sum to <= 1.0)
    imbalance_probability=0.10,
    bearing_fault_probability=0.10,

    # Fault onset: higher = more abrupt transition at health=0.85
    fault_onset_sharpness=20.0,

    # Flush error logs every N batches (guards against data loss on a crash)
    error_flush_interval=50,
)

generate = GenerateData(config, "data")
# generate.generate()
generate.combine_telemetry_to_single_parquet()