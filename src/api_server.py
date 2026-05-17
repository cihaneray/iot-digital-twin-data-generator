from __future__ import annotations

import json
import asyncio
import time
from datetime import timedelta
import random
import numpy as np
from dateutil import parser
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import websockets

from DataGeneration import GenerateData
from DataGeneration.Types import GlobalSimulationConfig

app = FastAPI()

def _parse_config(payload: dict) -> GlobalSimulationConfig:
    required_keys = [
        "seed", "date_from", "date_to", "telemetry_batch_frequency_seconds",
        "machine_count", "active_machines_per_batch", "cycle_length_min", "cycle_length_max",
    ]
    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")

    config_kwargs = {
        "seed": int(payload["seed"]),
        "date_from": parser.parse(payload["date_from"]),
        "date_to": parser.parse(payload["date_to"]),
        "telemetry_batch_frequency": timedelta(seconds=int(payload["telemetry_batch_frequency_seconds"])),
        "machine_count": int(payload["machine_count"]),
        "active_machines_per_batch": int(payload["active_machines_per_batch"]),
        "cycle_length_min": int(payload["cycle_length_min"]),
        "cycle_length_max": int(payload["cycle_length_max"]),
    }

    optional_keys = {
        "stress_multiplier": float, "failure_threshold": float, "ambient_temp_mean": float,
        "ambient_temp_amplitude": float, "imbalance_probability": float, "bearing_fault_probability": float,
        "fault_onset_sharpness": float, "fault_activation_health_threshold": float,
        "vibration_full_signature_health": float, "fault_vibration_speed_exponent": float,
        "fault_vibration_min_speed_ratio": float, "fault_vibration_load_influence": float,
        "fault_vibration_health_exponent": float, "harmonic_rolloff_exponent": float,
        "harmonic_resonance_strength": float, "base_signal_rms_nominal": float,
        "signal_rms_budget": float, "bearing_shock_min_speed_ratio": float,
        "bearing_shock_transition_ratio": float, "bearing_shock_load_influence": float,
        "bearing_shock_base_magnitude": float, "max_temperature": float, "min_pressure": float,
        "nominal_speed": float, "load_base": float, "pressure_base": float, "voltage_base": float,
        "vibration_sample_rate": int, "vibration_noise_level": float, "parquet_engine": str,
        "error_flush_interval": int,
    }

    for key, caster in optional_keys.items():
        if key in payload:
            config_kwargs[key] = caster(payload[key])

    return GlobalSimulationConfig(**config_kwargs)

@app.websocket("/ws/generate")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Standard Client Connected")
    # Target URI for forwarding data
    K8S_URI = "ws://localhost:8000/ws/kubernetes"

    try:
        # Step 1: Wait for config
        try:
            config_data = await websocket.receive_json()
            print(f"Config received: {config_data.get('seed')}")
        except Exception as e:
            print(f"Error receiving config: {type(e).__name__}: {e}")
            return

        try:
            config = _parse_config(config_data)
        except Exception as e:
            print(f"Config parsing error: {type(e).__name__}: {e}")
            await websocket.send_json({"error": f"Config error: {str(e)}"})
            return

        tick_seconds = float(config_data.get("tick_seconds", 1.0))

        # Step 2: Initialize generator and machines
        try:
            generator = GenerateData(config)
            machines = generator.create_machines(config.machine_count)
        except Exception as e:
            print(f"Initialization error: {type(e).__name__}: {e}")
            await websocket.send_json({"error": f"Init error: {str(e)}"})
            return

        schedules = {
            machine.name: {
                "speed_schedule": [float(s) for s in generator._generate_mission_profile(
                    random.randint(config.cycle_length_min, config.cycle_length_max) * 60
                ).tolist()],
                "index": 0,
                "cycle_length_minutes": random.randint(config.cycle_length_min, config.cycle_length_max)
            }
            for machine in machines
        }

        current_time = config.date_from

        # Step 3: Stream data and forward to K8S_URI
        print(f"Attempting to connect to K8S_URI: {K8S_URI}")

        k8s_ws = None
        try:
            # Add a timeout to connection attempt
            k8s_ws = await asyncio.wait_for(websockets.connect(K8S_URI), timeout=5.0)
            print("Successfully connected to K8S_URI")
        except Exception as e:
            print(f"Warning: Could not connect to K8S_URI ({type(e).__name__}: {e}). Proceeding with local stream only.")

        try:
            while True:
                active = generator.sample(machines, config.active_machines_per_batch)
                for machine in active:
                    sched = schedules[machine.name]
                    if sched["index"] >= len(sched["speed_schedule"]):
                        sched["cycle_length_minutes"] = random.randint(config.cycle_length_min, config.cycle_length_max)
                        sched["speed_schedule"] = [float(s) for s in generator._generate_mission_profile(
                            sched["cycle_length_minutes"] * 60
                        ).tolist()]
                        sched["index"] = 0

                    machine.set_speed(sched["speed_schedule"][sched["index"]])
                    sched["index"] += 1
                    machine.ambient_temperature = generator.get_ambient_temperature(current_time)

                    failure_timestamp = generator._estimate_failure_time(
                        machine, current_time, config.date_to, sched["cycle_length_minutes"],
                        initial_speed_schedule=np.array(sched["speed_schedule"][sched["index"]:])
                    )
                    rul_seconds = max(0.0, failure_timestamp - current_time.timestamp()) if failure_timestamp else None

                    state = machine.next_state(rul_seconds=rul_seconds)
                    # Mapping local field name 'vibration_intensity' to receiver's expected 'vibrationLevel'
                    state["vibrationLevel"] = state.pop("vibration_intensity", 0.0)
                    state.pop("vibration_pcm", None)
                    state["timestamp"] = current_time.isoformat()

                    # Send to the requesting client
                    try:
                        await websocket.send_json(state)
                    except Exception as e:
                        print(f"Client send failed: {type(e).__name__}")
                        return # Exit if the main client is gone

                    # Forward to the kubernetes endpoint
                    if k8s_ws:
                        try:
                            await k8s_ws.send(json.dumps(state))
                        except Exception as e:
                            print(f"K8S forward failed: {type(e).__name__}")
                            k8s_ws = None # Stop trying but keep main stream

                current_time += timedelta(seconds=tick_seconds)
                await asyncio.sleep(tick_seconds)
        finally:
            if k8s_ws:
                try:
                    await k8s_ws.close()
                except:
                    pass

    except WebSocketDisconnect:
        print("Standard Client Disconnected")
    except Exception as e:
        import traceback
        print(f"Stream interrupted: {type(e).__name__}: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8001)
