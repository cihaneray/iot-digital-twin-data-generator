import asyncio
import websockets
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def connect_to_generation_service(config_path: str, uri: str = "ws://localhost:8000/ws/generate"):
    """
    Connects to the WebSocket data generation service, sends a configuration,
    and processes the streaming data.
    """
    try:
        # Load the configuration from the provided JSON file
        with open(config_path, 'r') as f:
            config = json.load(f)

        async with websockets.connect(uri) as websocket:
            logger.info(f"Connected to {uri}")

            # Step 1: Send the configuration to start generation
            await websocket.send(json.dumps(config))
            logger.info("Configuration sent. Starting data reception...")

            # Step 2: Receive and process data in a loop
            try:
                while True:
                    message = await websocket.recv()
                    data = json.loads(message)

                    if "error" in data:
                        logger.error(f"Error from server: {data['error']}")
                        break

                    # Example processing: Just log the machine ID and RUL
                    # In a real FastAPI app, you might emit this to another frontend websocket
                    # or save it to a database.
                    machine_id = data.get("machineID")
                    rul = data.get("rul_seconds")
                    timestamp = data.get("timestamp")

                    logger.info(f"[{timestamp}] Machine: {machine_id} | RUL: {rul:.2f}s" if rul is not None else f"[{timestamp}] Machine: {machine_id} | RUL: N/A")

            except websockets.exceptions.ConnectionClosed:
                logger.info("Connection closed by server.")

    except FileNotFoundError:
        logger.error(f"Config file not found at {config_path}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    # Example usage:
    # Ensure the api_server.py is running on localhost:8000
    config_file = "configs/realtime_config.json"
    asyncio.run(connect_to_generation_service(config_file))

