from datetime import timedelta
from dateutil import parser
from DataGeneration import SimulationParameters, SensorParameters, GenerateData

sensor_parameters = SensorParameters(
    t_max=10000,
    d=0.05,
    a=-0.3,
    b=0.2,
    th=0.45
)

simulation_parameters = SimulationParameters(
    seed=42,
    date_from=parser.parse("Jan 01 2026 12:00AM"),
    date_to=parser.parse("Feb 21 2026 12:00AM"),
    telemetry_batch_frequency=timedelta(hours=1),
    telemetry_interval=timedelta(seconds=1),
    machine_count=1000,
    active_machines_per_batch=33,
    cycle_length_min=1,
    cycle_length_max=5
)

generate = GenerateData(simulation_parameters, sensor_parameters, '..\\data')

generate.generate()
