import asyncio
import json
import logging
import os
import datetime
from typing import Dict

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

from src.DataGeneration.Types import GlobalSimulationConfig
from src.DataGeneration.RotationalMachine import RotationalMachine

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SimulationServer")

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

# Global state for multi-tenancy
active_simulations: Dict[str, asyncio.Task] = {}
simulation_cancel_events: Dict[str, asyncio.Event] = {}

producer: AIOKafkaProducer = None


async def simulate_client_machines(client_id: str, config_data: dict, cancel_event: asyncio.Event):
    tick_rate = float(config_data.get("tick_rate", 1.0))
    machine_count = int(config_data.get("machine_count", 1))

    logger.info(f"[START] Client {client_id} için {machine_count} makine ile simülasyon başlatıldı. Tick rate: {tick_rate}s")

    # Initialize simulation config with dummy dates to satisfy constraints
    now = datetime.datetime.now(datetime.timezone.utc)
    date_from = now
    date_to = now + datetime.timedelta(days=365)
    
    # We use a deterministic but unique seed per client for variation if needed
    client_hash = hash(client_id) % (2**32)
    
    # Base kwargs for configuration
    config_kwargs = {
        "seed": client_hash,
        "date_from": date_from,
        "date_to": date_to,
        "telemetry_batch_frequency": datetime.timedelta(hours=1),
        "machine_count": machine_count,
        "active_machines_per_batch": machine_count,
        "cycle_length_min": 10,
        "cycle_length_max": 20
    }
    
    # Allow dynamic override of any physical/simulation parameter
    allowed_overrides = [
        "stress_multiplier", "failure_threshold", "ambient_temp_mean", "ambient_temp_amplitude",
        "imbalance_probability", "bearing_fault_probability", "fault_onset_sharpness", 
        "fault_activation_health_threshold", "vibration_full_signature_health", 
        "fault_vibration_speed_exponent", "fault_vibration_min_speed_ratio", 
        "fault_vibration_load_influence", "fault_vibration_health_exponent", 
        "harmonic_rolloff_exponent", "harmonic_resonance_strength", "base_signal_rms_nominal", 
        "signal_rms_budget", "bearing_shock_min_speed_ratio", "bearing_shock_transition_ratio",
        "bearing_shock_load_influence", "bearing_shock_base_magnitude", "max_temperature", 
        "min_pressure", "nominal_speed", "load_base", "pressure_base", "voltage_base", 
        "vibration_sample_rate", "vibration_noise_level", "cycle_length_min", "cycle_length_max"
    ]
    
    for key in allowed_overrides:
        if key in config_data:
            config_kwargs[key] = config_data[key]
            
    sim_config = GlobalSimulationConfig(**config_kwargs)

    machines = [RotationalMachine(f"M_{i:04d}", sim_config) for i in range(1, machine_count + 1)]
    
    # Pre-set speed so wear logic activates
    for m in machines:
        m.set_speed(1000.0)

    try:
        while not cancel_event.is_set():
            current_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
            
            for machine in machines:
                try:
                    # Update simulation state
                    state = machine.next_state(rul_seconds=None)
                    
                    # Extract and format vibration PCM
                    vibration_pcm = state.pop("vibration_pcm")
                    if "timestamp" in state:
                        del state["timestamp"]
                        
                    # 1. Produce Telemetry
                    telemetry_msg = {
                        "client_id": client_id,
                        "machine_id": machine.name,
                        "timestamp": current_time,
                        "metrics": state
                    }
                    await producer.send_and_wait("simulation-telemetry", json.dumps(telemetry_msg).encode("utf-8"))
                    
                    # 2. Produce Vibration
                    vibration_msg = {
                        "client_id": client_id,
                        "machine_id": machine.name,
                        "timestamp": current_time,
                        "vibration_signal": vibration_pcm.tolist()
                    }
                    await producer.send_and_wait("simulation-vibration", json.dumps(vibration_msg).encode("utf-8"))
                    
                    # 3. Produce Warning Logs (If fault just got active or is active)
                    if state.get("fault_active") == 1:
                        log_msg = {
                            "client_id": client_id,
                            "machine_id": machine.name,
                            "timestamp": current_time,
                            "level": "WARNING",
                            "code": f"Anomaly Detected: {state.get('fault_type')}"
                        }
                        await producer.send_and_wait("simulation-logs", json.dumps(log_msg).encode("utf-8"))

                except Exception as e:
                    # Over-Temp or Mechanical Failure exceptions raised from next_state()
                    error_msg = str(e)
                    log_msg = {
                        "client_id": client_id,
                        "machine_id": machine.name,
                        "timestamp": current_time,
                        "level": "CRITICAL",
                        "code": error_msg
                    }
                    await producer.send_and_wait("simulation-logs", json.dumps(log_msg).encode("utf-8"))
                    
                    # Reset machine to simulate a 'fixed' state so it continues generating data
                    machine.__reset__()
                    machine.set_speed(1000.0)
                    fixed_log_msg = {
                        "client_id": client_id,
                        "machine_id": machine.name,
                        "timestamp": current_time,
                        "level": "INFO",
                        "code": "fixed"
                    }
                    await producer.send_and_wait("simulation-logs", json.dumps(fixed_log_msg).encode("utf-8"))

            await asyncio.sleep(tick_rate)

    except asyncio.CancelledError:
        # Task was cancelled gracefully
        pass
    except Exception as e:
        logger.error(f"[ERROR] Client {client_id} simülasyonunda beklenmedik hata: {e}")
    finally:
        logger.info(f"[STOP] Client {client_id} için simülasyon görevden alındı/durduruldu.")


async def process_command(msg_value: bytes):
    try:
        data = json.loads(msg_value.decode("utf-8"))
        client_id = data.get("client_id")
        action = data.get("action")

        if not client_id or not action:
            logger.warning("[WARNING] Eksik client_id veya action alındı.")
            return

        if action == "START":
            config = data.get("config", {})
            if client_id in active_simulations:
                logger.warning(f"[WARNING] Client {client_id} zaten aktif. Görev yeniden başlatılıyor...")
                simulation_cancel_events[client_id].set()
            
            cancel_event = asyncio.Event()
            simulation_cancel_events[client_id] = cancel_event
            task = asyncio.create_task(simulate_client_machines(client_id, config, cancel_event))
            active_simulations[client_id] = task

        elif action == "STOP":
            if client_id in active_simulations:
                simulation_cancel_events[client_id].set()
                del simulation_cancel_events[client_id]
                del active_simulations[client_id]
            else:
                logger.warning(f"[WARNING] STOP komutu alındı ama Client {client_id} sistemde aktif değil.")
        else:
            logger.warning(f"[WARNING] Bilinmeyen action: {action}")

    except json.JSONDecodeError:
        logger.error("[ERROR] Gelen mesaj geçerli bir JSON değil.")
    except Exception as e:
        logger.error(f"[ERROR] Komut işlenirken hata oluştu: {e}")


async def main():
    global producer

    logger.info(f"Kafka Bootstrap Servers: {KAFKA_BOOTSTRAP_SERVERS}")

    consumer = AIOKafkaConsumer(
        "simulation-commands",
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id="simulation_server_group",
        value_deserializer=lambda m: m
    )

    producer = AIOKafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS
    )

    try:
        # Start Kafka clients
        await consumer.start()
        await producer.start()
        logger.info("Kafka Consumer ve Producer başarıyla başlatıldı. 'simulation-commands' dinleniyor...")

        async for msg in consumer:
            logger.info(f"[KOMUT ALINDI] Topic: {msg.topic}, Partition: {msg.partition}, Offset: {msg.offset}")
            await process_command(msg.value)

    except Exception as e:
        logger.error(f"[ERROR] Kafka bağlantısı koptu veya başlatılamadı: {e}")
    finally:
        # Graceful shutdown
        await consumer.stop()
        if producer:
            await producer.stop()
        
        # Stop all active simulations
        for client_id, event in simulation_cancel_events.items():
            event.set()
        
        if active_simulations:
            await asyncio.gather(*active_simulations.values(), return_exceptions=True)
            
        logger.info("Sunucu güvenli bir şekilde kapatıldı.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Kullanıcı tarafından manuel durduruldu (KeyboardInterrupt).")
