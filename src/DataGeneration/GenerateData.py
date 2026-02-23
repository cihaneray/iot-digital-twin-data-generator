import datetime
import math
import random
import pandas as pd
from tqdm import tqdm

from .RotationalMachine import RotationalMachine
from .Types import SimulationParameters, SensorParameters


class GenerateData:
    def __init__(self, simulation_parameters: SimulationParameters, sensor_parameters: SensorParameters,
                 data_dir: str = 'data') -> None:
        self.data_dir = data_dir
        self.simulation_parameters = simulation_parameters
        self.sensor_parameters = sensor_parameters
        random.seed(self.simulation_parameters.seed)

    @staticmethod
    def create_machines(n):
        """Initializes machines without external health iterators."""
        return [RotationalMachine('M_{0:04d}'.format(i)) for i in range(n)]

    @staticmethod
    def h_generator(ttf, d_val, a_val, b_val, th_val=0.0):
        for remain_time in range(ttf, -1, -1):
            h_val = 1 - d_val - math.exp(a_val * remain_time ** b_val)
            if h_val < th_val:
                break
            yield remain_time, h_val

    @staticmethod
    def sample(machines, m):
        return [machines[i] for i in sorted(random.sample(range(len(machines)), m))]

    def generate(self):
        machines = self.create_machines(self.simulation_parameters.machine_count)
        telemetry = []
        errors = []

        date_interval = (self.simulation_parameters.date_to - self.simulation_parameters.date_from)
        max_count = date_interval // self.simulation_parameters.telemetry_batch_frequency

        with tqdm(total=max_count, desc="Data Generation:") as pbar:
            while self.simulation_parameters.date_from + self.simulation_parameters.telemetry_batch_frequency < self.simulation_parameters.date_to:
                s = self.sample(machines, self.simulation_parameters.active_machines_per_batch)
                telemetry.clear()

                for m in s:
                    if m.broken:
                        # Repair: Reset the specific machine instance
                        m.__init__(m.name)
                        errors.append({
                            'timestamp': self.simulation_parameters.date_from,
                            'machineID': m.name,
                            'level': 'INFO',
                            'code': 'fixed'
                        })
                        continue

                    # Determine duty cycle length
                    l = random.randint(self.simulation_parameters.cycle_length_min,
                                       self.simulation_parameters.cycle_length_max)
                    offset = random.randint(0, 60 - l)

                    m.set_speed(1000)
                    duration = l * 60
                    cooldown_point = duration - 20

                    for i in range(duration):
                        if i == cooldown_point:
                            m.set_speed(0)

                        ts = self.simulation_parameters.date_from + datetime.timedelta(seconds=offset * 60 + i)
                        try:
                            state = m.next_state()
                            state['timestamp'] = ts
                            state['machineID'] = m.name
                            telemetry.append(state)

                            # If the machine naturally slowed to 0 after cooldown
                            if not state['speed'] and i >= cooldown_point:
                                break
                        except Exception as e:
                            errors.append({
                                'timestamp': ts,
                                'machineID': m.name,
                                'level': 'CRITICAL',
                                'code': str(e)
                            })
                            break

                if telemetry:
                    # Parquet export - vibration is excluded to save space
                    telemetry_df = pd.DataFrame(telemetry).drop('vibration', axis=1, errors='ignore')
                    telemetry_df.index = telemetry_df['timestamp']
                    del telemetry_df['timestamp']

                    suffix = self.simulation_parameters.date_from.strftime("%Y%m%d-%H%M%S")
                    telemetry_df.to_parquet('{0}/telemetry/telemetry-{1}.parquet'.format(self.data_dir, suffix),
                                            engine='fastparquet',
                                            times='int96')
                    del telemetry_df

                self.simulation_parameters.date_from += self.simulation_parameters.telemetry_batch_frequency
                pbar.update(1)

        if errors:
            logs_df = pd.DataFrame(errors)
            sequence_count = len(logs_df[logs_df.level == 'CRITICAL'])
            logs_df.index = logs_df['timestamp']
            del logs_df['timestamp']
            logs_df.to_parquet(self.data_dir + '/logs/logs.parquet', engine='fastparquet', times='int96')
            print('{0} failure event(s) recorded.'.format(sequence_count))
