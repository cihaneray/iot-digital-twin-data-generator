from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import timedelta
import random
import numpy as np
from pathlib import Path

from dateutil import parser

from DataGeneration import GenerateData
from DataGeneration.Types import GlobalSimulationConfig


@dataclass
class _MachineSchedule:
    speed_schedule: list[float]
    index: int
    cycle_length_minutes: int


def _new_schedule(generator: GenerateData, cycle_length_minutes: int) -> list[float]:
    schedule = generator._generate_mission_profile(cycle_length_minutes * 60)
    return [float(speed) for speed in schedule.tolist()]


def _load_config(config_path: Path) -> GlobalSimulationConfig:
    payload = json.loads(config_path.read_text(encoding="utf-8"))

    required_keys = [
        "seed",
        "date_from",
        "date_to",
        "telemetry_batch_frequency_seconds",
        "machine_count",
        "active_machines_per_batch",
        "cycle_length_min",
        "cycle_length_max",
    ]
    missing = [key for key in required_keys if key not in payload]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Missing required config keys: {missing_text}")

    config_kwargs = {
        "seed": int(payload["seed"]),
        "date_from": parser.parse(payload["date_from"]),
        "date_to": parser.parse(payload["date_to"]),
        "telemetry_batch_frequency": timedelta(
            seconds=int(payload["telemetry_batch_frequency_seconds"])
        ),
        "machine_count": int(payload["machine_count"]),
        "active_machines_per_batch": int(payload["active_machines_per_batch"]),
        "cycle_length_min": int(payload["cycle_length_min"]),
        "cycle_length_max": int(payload["cycle_length_max"]),
    }

    optional_keys = {
        "stress_multiplier": float,
        "failure_threshold": float,
        "ambient_temp_mean": float,
        "ambient_temp_amplitude": float,
        "imbalance_probability": float,
        "bearing_fault_probability": float,
        "fault_onset_sharpness": float,
        "fault_activation_health_threshold": float,
        "vibration_full_signature_health": float,
        "fault_vibration_speed_exponent": float,
        "fault_vibration_min_speed_ratio": float,
        "fault_vibration_load_influence": float,
        "fault_vibration_health_exponent": float,
        "harmonic_rolloff_exponent": float,
        "harmonic_resonance_strength": float,
        "base_signal_rms_nominal": float,
        "signal_rms_budget": float,
        "bearing_shock_min_speed_ratio": float,
        "bearing_shock_transition_ratio": float,
        "bearing_shock_load_influence": float,
        "bearing_shock_base_magnitude": float,
        "max_temperature": float,
        "min_pressure": float,
        "nominal_speed": float,
        "load_base": float,
        "pressure_base": float,
        "voltage_base": float,
        "vibration_sample_rate": int,
        "vibration_noise_level": float,
        "parquet_engine": str,
        "error_flush_interval": int,
    }

    for key, caster in optional_keys.items():
        if key in payload:
            config_kwargs[key] = caster(payload[key])

    return GlobalSimulationConfig(**config_kwargs)


def generate_realtime(
    config: GlobalSimulationConfig,
    tick_seconds: float,
    max_runtime: int | None,
) -> int:
    generator = GenerateData(config)
    machines = generator.create_machines(config.machine_count)
    schedules = {
        machine.name: _MachineSchedule(
            speed_schedule=_new_schedule(
                generator,
                random.randint(config.cycle_length_min, config.cycle_length_max),
            ),
            index=0,
            cycle_length_minutes=random.randint(
                config.cycle_length_min,
                config.cycle_length_max,
            ),
        )
        for machine in machines
    }

    total_rows = 0
    current_time = config.date_from
    end_time = None if max_runtime is None else time.perf_counter() + max_runtime

    while end_time is None or time.perf_counter() < end_time:
        tick_start = time.perf_counter()
        active = generator.sample(machines, config.active_machines_per_batch)

        for machine in active:
            schedule = schedules[machine.name]
            if schedule.index >= len(schedule.speed_schedule):
                schedule.cycle_length_minutes = random.randint(
                    config.cycle_length_min, config.cycle_length_max
                )
                schedule.speed_schedule = _new_schedule(
                    generator,
                    schedule.cycle_length_minutes,
                )
                schedule.index = 0

            machine.set_speed(schedule.speed_schedule[schedule.index])
            schedule.index += 1

            machine.ambient_temperature = generator.get_ambient_temperature(current_time)

            # Estimate failure time to calculate RUL
            failure_timestamp = generator._estimate_failure_time(
                machine,
                current_time,
                config.date_to,
                schedule.cycle_length_minutes,
                initial_speed_schedule=np.array(schedule.speed_schedule[schedule.index:]),
            )

            rul_seconds = None
            if failure_timestamp is not None:
                rul_seconds = max(0.0, failure_timestamp - current_time.timestamp())

            state = machine.next_state(rul_seconds=rul_seconds)
            state.pop("vibration_pcm", None)
            state["timestamp"] = current_time.isoformat()

            total_rows += 1
            print(json.dumps(state))

        current_time += timedelta(seconds=tick_seconds)
        elapsed = time.perf_counter() - tick_start
        if elapsed < tick_seconds:
            if end_time is None:
                sleep_for = tick_seconds - elapsed
            else:
                remaining = end_time - time.perf_counter()
                sleep_for = min(tick_seconds - elapsed, max(0.0, remaining))
            if sleep_for > 0.0:
                time.sleep(sleep_for)

    return total_rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate realtime telemetry for a fixed wall-clock duration without saving."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tick-seconds", type=float, default=1.0, help="Interval between data production cycles")
    parser.add_argument("--max-runtime", type=int, help="Maximum execution time in seconds. If omitted, runs indefinitely.")
    parser.add_argument("--continuous", action="store_true", help="Run indefinitely (ignores max-runtime if set)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config)
    runtime = None if args.continuous else args.max_runtime
    generate_realtime(config, args.tick_seconds, runtime)


if __name__ == "__main__":
    main()
