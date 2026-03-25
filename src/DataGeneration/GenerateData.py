import datetime
import math
import os
import random

import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import cast, Literal

from .RotationalMachine import RotationalMachine
from .Types import GlobalSimulationConfig


class GenerateData:
    def __init__(
            self,
            simulation_parameters: GlobalSimulationConfig,
            data_dir: str = "data",
    ):
        self.data_dir = data_dir
        self.telemetry_dir = os.path.join(data_dir, "telemetry")
        self.vibration_dir = os.path.join(data_dir, "vibration")
        self.logs_dir = os.path.join(data_dir, "logs")

        for directory in (self.telemetry_dir, self.vibration_dir, self.logs_dir):
            os.makedirs(directory, exist_ok=True)

        self.simulation_parameters = simulation_parameters
        random.seed(self.simulation_parameters.seed)
        self.active_faults_logged: set[str] = set()

    def create_machines(self, count: int) -> list[RotationalMachine]:
        return [
            RotationalMachine(f"M_{i:04d}", self.simulation_parameters)
            for i in range(count)
        ]

    @staticmethod
    def sample(machines: list, sample_size: int) -> list:
        return [machines[i] for i in sorted(random.sample(range(len(machines)), sample_size))]

    def get_ambient_temperature(self, current_time: datetime.datetime) -> float:
        seconds_in_day = 24 * 60 * 60
        elapsed = (
                current_time
                - current_time.replace(hour=0, minute=0, second=0, microsecond=0)
        ).total_seconds()
        day_fraction = elapsed / seconds_in_day
        phase_shift = 2 * math.pi * (day_fraction - 0.125)
        cycle_variation = -math.cos(phase_shift)
        return (
                self.simulation_parameters.ambient_temp_mean
                + self.simulation_parameters.ambient_temp_amplitude * cycle_variation
        )

    def _estimate_failure_time(
            self,
            machine: RotationalMachine,
            sim_start: datetime.datetime,
            sim_end: datetime.datetime,
            cycle_length: int,
            initial_speed_schedule: np.ndarray | None = None,
            first_batch_offset_seconds: int = 0,
    ) -> float | None:
        damage = machine.accumulated_damage
        threshold = machine.config.failure_threshold
        cfg = machine.config

        expected_cycle_length = max(
            1, int(round((cfg.cycle_length_min + cfg.cycle_length_max) / 2.0))
        )
        current_batch = sim_start
        is_first_batch = True

        while current_batch < sim_end:
            if is_first_batch and initial_speed_schedule is not None:
                speed_schedule = initial_speed_schedule
            else:
                duration = (cycle_length if is_first_batch else expected_cycle_length) * 60
                speed_schedule = self._generate_mission_profile(duration)

            batch_offset_seconds = first_batch_offset_seconds if is_first_batch else 0

            for second, speed in enumerate(speed_schedule):
                timestamp = current_batch + datetime.timedelta(
                    seconds=batch_offset_seconds + second
                )
                if timestamp >= sim_end:
                    return None

                speed_value = float(speed)
                if speed_value <= 0.1:
                    continue

                amb_temp = self.get_ambient_temperature(timestamp)
                h = max(0.001, 1.0 - damage / threshold)
                health_penalty = 1.0 + 0.5 * (1.0 - h)

                friction = 1.0
                if "bearing" in machine.active_faults:
                    friction *= 1.25
                if "imbalance" in machine.active_faults:
                    friction *= 1.1

                load = cfg.load_base * (speed_value / cfg.nominal_speed)
                load_penalty = 1.0 + 0.2 * (load / 100.0)
                approx_temp = (
                    amb_temp
                    + (speed_value / cfg.nominal_speed)
                    * 50.0
                    * health_penalty
                    * friction
                    * load_penalty
                )

                wear_delta = RotationalMachine.compute_wear_step(
                    stress_multiplier=cfg.stress_multiplier,
                    speed=speed_value,
                    load=load,
                    temperature=approx_temp,
                    active_faults=machine.active_faults,
                    accumulated_damage=damage,
                    nominal_speed=cfg.nominal_speed,
                )

                if wear_delta <= 0.0:
                    continue

                damage += wear_delta
                if damage >= threshold:
                    return timestamp.timestamp()

            current_batch += cfg.telemetry_batch_frequency
            is_first_batch = False

        return None

    def _handle_broken_machine(
            self, machine: RotationalMachine, current_time: datetime.datetime
    ) -> dict:
        machine.__reset__()
        self.active_faults_logged.discard(machine.name)
        return {
            "timestamp": current_time,
            "machineID": machine.name,
            "level": "INFO",
            "code": "fixed",
        }

    def _generate_mission_profile(self, duration_seconds: int) -> np.ndarray:
        if duration_seconds <= 60:
            schedule = np.full(duration_seconds, 1000.0)
            schedule[-20:] = 0.0
            return schedule

        schedule = np.zeros(duration_seconds)
        remaining = duration_seconds - 20
        idx = 0
        
        warmup = min(120, remaining)
        schedule[idx : idx + warmup] = 500.0
        idx += warmup
        remaining -= warmup
        
        speeds = [400.0, 800.0, 1000.0, 1200.0]
        weights = [0.2, 0.4, 0.3, 0.1]
        
        while remaining > 0:
            phase_len = random.randint(300, 1800)
            phase_len = min(phase_len, remaining)
            spd = random.choices(speeds, weights=weights, k=1)[0]
            schedule[idx : idx + phase_len] = spd
            idx += phase_len
            remaining -= phase_len
            
        schedule[-20:] = 0.0
        return schedule

    def _process_machine_cycle(
            self,
            machine: RotationalMachine,
            current_time: datetime.datetime,
            sim_end: datetime.datetime,
            cycle_length: int,
            offset: int,
    ) -> tuple[list[dict], dict[str, np.ndarray], list[dict]]:
        duration = cycle_length * 60
        speed_schedule = self._generate_mission_profile(duration)

        telemetry_rows: list[dict] = []
        vibration_arrays: dict[str, np.ndarray] = {}
        error_rows: list[dict] = []

        cycle_pcm_buffers: list[np.ndarray] = []

        # Initial failure estimate (before fault may activate)
        failure_timestamp = self._estimate_failure_time(
            machine,
            current_time,
            sim_end,
            cycle_length,
            initial_speed_schedule=speed_schedule,
            first_batch_offset_seconds=offset * 60,
        )

        for second in range(duration):
            machine.set_speed(float(speed_schedule[second]))

            timestamp = current_time + datetime.timedelta(seconds=offset * 60 + second)
            machine.ambient_temperature = (
                    self.get_ambient_temperature(timestamp) + random.uniform(-0.1, 0.1)
            )

            try:
                rul_seconds = (
                    (failure_timestamp - timestamp.timestamp())
                    if failure_timestamp is not None
                    else None
                )
                if rul_seconds is not None:
                    rul_seconds = max(0.0, rul_seconds)

                pre_fault = machine.fault_active

                state = machine.next_state(rul_seconds=rul_seconds)

                if machine.fault_active and not pre_fault:
                    failure_timestamp = self._estimate_failure_time(
                        machine,
                        timestamp,
                        sim_end,
                        cycle_length,
                        initial_speed_schedule=speed_schedule[second:],
                    )
                    if failure_timestamp is not None:
                        state["rul_seconds"] = max(
                            0.0, failure_timestamp - timestamp.timestamp()
                        )
                    else:
                        state["rul_seconds"] = None

                if state["fault_active"] and machine.name not in self.active_faults_logged:
                    error_rows.append({
                        "timestamp": timestamp,
                        "machineID": machine.name,
                        "level": "WARNING",
                        "code": f"Anomaly Detected: {machine.fault_type}",
                    })
                    self.active_faults_logged.add(machine.name)

                pcm_array: np.ndarray = state.pop("vibration_pcm")
                cycle_pcm_buffers.append(pcm_array)

                state["timestamp"] = timestamp
                telemetry_rows.append(state)

                if state["speed"] < 1.0 and second >= duration - 20:
                    break

            except Exception as exc:
                error_rows.append({
                    "timestamp": timestamp,
                    "machineID": machine.name,
                    "level": "CRITICAL",
                    "code": str(exc),
                })
                break

        if cycle_pcm_buffers:
            cycle_key = f"{machine.name}_{current_time.strftime('%Y%m%dT%H%M%S')}"
            vibration_arrays[cycle_key] = np.concatenate(cycle_pcm_buffers)

        return telemetry_rows, vibration_arrays, error_rows

    def _save_telemetry(self, rows: list[dict], suffix: str):
        if not rows:
            return
        df = pd.DataFrame(rows)
        df["fault_type"] = df["fault_type"].astype("category")
        df.set_index("timestamp", inplace=True)
        engine = cast(Literal["auto", "pyarrow", "fastparquet"], self.simulation_parameters.parquet_engine)
        df.to_parquet(
            os.path.join(self.telemetry_dir, f"telemetry-{suffix}.parquet"),
            engine=engine,
        )

    def _save_vibration(self, vibration_arrays: dict[str, np.ndarray], suffix: str):
        if not vibration_arrays:
            return
        path = os.path.join(self.vibration_dir, f"vibration-{suffix}.npz")
        np.savez_compressed(path, **vibration_arrays)

    def _flush_errors(self, errors: list[dict], append: bool = False):
        if not errors:
            return

        engine = cast(Literal["auto", "pyarrow", "fastparquet"], self.simulation_parameters.parquet_engine)
        log_path = os.path.join(self.logs_dir, "logs.parquet")

        new_df = pd.DataFrame(errors)
        new_df.set_index("timestamp", inplace=True)

        if append and os.path.exists(log_path):
            existing_df = pd.read_parquet(log_path, engine=engine)
            combined_df = pd.concat([existing_df, new_df])
            combined_df.to_parquet(log_path, engine=engine)
        else:
            new_df.to_parquet(log_path, engine=engine)

        errors.clear()

    def generate(self):
        cfg = self.simulation_parameters
        machines = self.create_machines(cfg.machine_count)
        errors: list[dict] = []

        current_time: datetime.datetime = cfg.date_from
        date_interval = cfg.date_to - cfg.date_from
        max_count = date_interval // cfg.telemetry_batch_frequency
        batch_number = 0

        with tqdm(total=max_count, desc="Data Generation") as progress_bar:
            while current_time + cfg.telemetry_batch_frequency < cfg.date_to:
                sampled_machines = self.sample(machines, cfg.active_machines_per_batch)

                batch_telemetry: list[dict] = []
                batch_vibration: dict[str, np.ndarray] = {}

                for machine in sampled_machines:
                    if machine.broken:
                        errors.append(
                            self._handle_broken_machine(machine, current_time)
                        )
                        continue

                    cycle_length = random.randint(cfg.cycle_length_min, cfg.cycle_length_max)
                    offset = random.randint(0, 60 - cycle_length)

                    tel, vib, errs = self._process_machine_cycle(
                        machine, current_time, cfg.date_to, cycle_length, offset
                    )
                    batch_telemetry.extend(tel)
                    batch_vibration.update(vib)
                    errors.extend(errs)

                suffix = current_time.strftime("%Y%m%d-%H%M%S")
                self._save_telemetry(batch_telemetry, suffix)
                self._save_vibration(batch_vibration, suffix)

                batch_number += 1
                if batch_number % cfg.error_flush_interval == 0:
                    self._flush_errors(
                        errors,
                        append=(batch_number > cfg.error_flush_interval),
                    )

                current_time += cfg.telemetry_batch_frequency
                progress_bar.update(1)

        log_path = os.path.join(self.logs_dir, "logs.parquet")
        self._flush_errors(errors, append=os.path.exists(log_path))

        if os.path.exists(log_path):
            logs_df = pd.read_parquet(
                log_path, engine=cfg.parquet_engine
            )
            failure_count = (logs_df["level"] == "CRITICAL").sum()
            print(f"{failure_count} total failure(s) recorded in logs.")
        else:
            print("No events logged.")
