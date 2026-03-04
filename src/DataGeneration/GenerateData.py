import datetime
import math
import os
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

from .RotationalMachine import RotationalMachine
from .Types import GlobalSimulationConfig


class GenerateData:
    """
    Orchestrates the multi-machine time-series simulation and writes outputs to disk.

    Output layout
    -------------
    data/
        telemetry/  → one parquet per batch (scalar sensor readings)
                      columns: machineID, speed_desired, speed, temperature, pressure,
                               ambient_temperature, health_index, fault_type,
                               fault_active, rul_seconds
        vibration/  → one .npz per batch (raw int16 PCM arrays)
                      keys: "<machineID>_<ISO timestamp>"
        logs/       → single parquet (WARNING / CRITICAL / INFO events)
    """

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

        # Seed module-level random only for sampling / ambient jitter
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
        """Sinusoidal 24-hour ambient temperature cycle peaking at ~15:00."""
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

    @staticmethod
    def _estimate_failure_time(
            machine: RotationalMachine,
            sim_start: datetime.datetime,
            sim_end: datetime.datetime,
            cycle_length: int,
    ) -> float | None:
        """
        Estimates the Unix timestamp at which `machine` will reach health_index=0,
        by replaying damage accumulation with the same parameters used during the
        live simulation cycle.

        This is intentionally a *lightweight* estimate: it uses the same
        `_calculate_wear_delta` logic but skips vibration generation and I/O.
        The returned value is a float (seconds since epoch) or None if the machine
        does not fail within the remaining simulation window.

        The caller stores this value and passes it to `next_state(failure_time=...)`
        so that `rul_seconds` in each telemetry row equals
        `failure_timestamp - current_timestamp`.
        """
        # Snapshot the machine's current damage state
        damage_snapshot = machine.accumulated_damage
        threshold = machine.config.failure_threshold

        duration = cycle_length * 60
        cooldown_point = duration - 20

        current_time = sim_start
        second = 0

        # Shadow-simulate until failure or end of a window
        while current_time < sim_end:
            cycle_second = second % duration
            speed = 0.0 if cycle_second >= cooldown_point else 1000.0

            approx_health = max(0.0, 1.0 - damage_snapshot / threshold)
            health_penalty = 1.0 + 0.5 * (1.0 - approx_health)
            friction_heat = 1.25 if (machine.fault_active and machine.fault_type == "bearing") else 1.0
            approx_temp = (
                    machine.ambient_temperature
                    + (speed / 1000.0) * 53.3 * health_penalty * friction_heat
            )

            wear_delta = RotationalMachine.compute_wear_step(
                stress_multiplier=machine.config.stress_multiplier,
                speed=speed,
                temperature=approx_temp,
                fault_active=machine.fault_active,  # Assuming the fault state persists in estimation
                fault_type=machine.fault_type,
                accumulated_damage=damage_snapshot
            )
            damage_snapshot += wear_delta

            if damage_snapshot >= threshold:
                return current_time.timestamp()

            current_time += datetime.timedelta(seconds=1)
            second += 1

        return None  # Machine survives the full simulation window

    def _handle_broken_machine(
            self, machine: RotationalMachine, current_time: datetime.datetime
    ) -> dict:
        machine.reset()
        self.active_faults_logged.discard(machine.name)
        return {
            "timestamp": current_time,
            "machineID": machine.name,
            "level": "INFO",
            "code": "fixed",
        }

    def _process_machine_cycle(
            self,
            machine: RotationalMachine,
            current_time: datetime.datetime,
            sim_end: datetime.datetime,
            cycle_length: int,
            offset: int,
    ) -> tuple[list[dict], dict[str, np.ndarray], list[dict]]:
        """
        Simulates one operational cycle for a single machine.

        Returns
        -------
        telemetry_rows  : list of scalar-sensor dicts (one per second)
        vibration_arrays: {key_str → int16 PCM array}
        error_rows      : list of log event dicts
        """
        machine.set_speed(1000)
        duration = cycle_length * 60
        cooldown_point = duration - 20

        telemetry_rows: list[dict] = []
        vibration_arrays: dict[str, np.ndarray] = {}
        error_rows: list[dict] = []

        failure_timestamp = self._estimate_failure_time(
            machine, current_time, sim_end, cycle_length
        )

        for second in range(duration):
            if second == cooldown_point:
                machine.set_speed(0)

            timestamp = current_time + datetime.timedelta(seconds=offset * 60 + second)

            machine.ambient_temperature = (
                    self.get_ambient_temperature(timestamp) + random.uniform(-0.1, 0.1)
            )

            try:
                # Compute rul_seconds for this specific timestamp
                rul_seconds = (
                    (failure_timestamp - timestamp.timestamp())
                    if failure_timestamp is not None
                    else None
                )
                # Clamp: machine cannot have negative RUL
                if rul_seconds is not None:
                    rul_seconds = max(0.0, rul_seconds)

                state = machine.next_state(rul_seconds=rul_seconds)

                if state["fault_active"] and machine.name not in self.active_faults_logged:
                    error_rows.append({
                        "timestamp": timestamp,
                        "machineID": machine.name,
                        "level": "WARNING",
                        "code": f"Anomaly Detected: {machine.fault_type}",
                    })
                    self.active_faults_logged.add(machine.name)

                pcm_array: np.ndarray = state.pop("vibration_pcm")
                ts_key = f"{machine.name}_{timestamp.strftime('%Y%m%dT%H%M%S')}"
                vibration_arrays[ts_key] = pcm_array

                state["timestamp"] = timestamp
                telemetry_rows.append(state)

                # Early exit once machine has cooled and stopped
                if not state["speed"] and second >= cooldown_point:
                    break

            except Exception as exc:
                error_rows.append({
                    "timestamp": timestamp,
                    "machineID": machine.name,
                    "level": "CRITICAL",
                    "code": str(exc),
                })
                break

        return telemetry_rows, vibration_arrays, error_rows

    def _save_telemetry(self, rows: list[dict], suffix: str):
        if not rows:
            return
        df = pd.DataFrame(rows)
        df["fault_type"] = df["fault_type"].astype("category")
        df.set_index("timestamp", inplace=True)
        df.to_parquet(
            os.path.join(self.telemetry_dir, f"telemetry-{suffix}.parquet"),
            engine="fastparquet",
        )

    def _save_vibration(self, vibration_arrays: dict[str, np.ndarray], suffix: str):
        """
        Saves all vibration PCM arrays for a batch as a single compressed .npz file.

        Keys are "<machineID>_<ISO timestamp>" strings.
        Loading:
            data = np.load('vibration-...npz')
            signal = data['M_0001_20260101T120000']
        """
        if not vibration_arrays:
            return
        path = os.path.join(self.vibration_dir, f"vibration-{suffix}.npz")
        np.savez_compressed(path, **vibration_arrays)

    def _flush_errors(self, errors: list[dict], append: bool = False):
        """
        Writes the current error buffer to disk and clears it in-place.
        Uses append mode (read-modify-write) when `append=True` so partial
        runs are not lost.
        """
        if not errors:
            return

        new_df = pd.DataFrame(errors)
        new_df.set_index("timestamp", inplace=True)

        log_path = os.path.join(self.logs_dir, "logs.parquet")

        if append and os.path.exists(log_path):
            existing_df = pd.read_parquet(log_path, engine="fastparquet")
            combined_df = pd.concat([existing_df, new_df])
            combined_df.to_parquet(log_path, engine="fastparquet")
        else:
            new_df.to_parquet(log_path, engine="fastparquet")

        errors.clear()

    def generate(self):
        """
        Runs the full simulation and writes telemetry, vibration, and logs to disk.
        """
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
                        machine,
                        current_time,
                        cfg.date_to,  # pass simulation end for RUL horizon
                        cycle_length,
                        offset,
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

                # FIX #5: advance local clock, not the config object
                current_time += cfg.telemetry_batch_frequency
                progress_bar.update(1)

        log_path = os.path.join(self.logs_dir, "logs.parquet")
        self._flush_errors(errors, append=os.path.exists(log_path))

        if os.path.exists(log_path):
            logs_df = pd.read_parquet(log_path, engine="fastparquet")
            failure_count = (logs_df["level"] == "CRITICAL").sum()
            print(f"{failure_count} total failure(s) recorded in logs.")
        else:
            print("No events logged.")
