from datetime import timedelta
from dateutil import parser
from DataGeneration import GlobalSimulationConfig, GenerateData

config = GlobalSimulationConfig(
    seed=42,
    date_from=parser.parse("Jan 01 2026 12:00AM"),
    date_to=parser.parse("Feb 21 2026 12:00AM"),
    telemetry_batch_frequency=timedelta(hours=1),
    telemetry_interval=timedelta(seconds=1),
    machine_count=1000,
    active_machines_per_batch=33,
    cycle_length_min=1,
    cycle_length_max=5,
    stress_multiplier=1.5,      # Machines wear out 50% faster
    failure_threshold=80.0     # Machines are slightly more durable
)

# GenerateData.py remains largely the same but passes 'config' to machines
generate = GenerateData(config, 'data')
generate.generate()