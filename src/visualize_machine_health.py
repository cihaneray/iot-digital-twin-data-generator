import pandas as pd
import matplotlib.pyplot as plt
import glob
import os


def visualize_machine_health(machine_id, data_dir='data/telemetry'):
    # Load all telemetry files
    files = glob.glob(os.path.join(data_dir, "*.parquet"))
    if not files:
        print("No telemetry files found.")
        return

    all_data = []
    for f in files:
        df = pd.read_parquet(f)
        # Filter for the specific machine
        m_data = df[df['machineID'] == machine_id]
        all_data.append(m_data)

    df_total = pd.concat(all_data).sort_index()

    if df_total.empty:
        print(f"No data found for machine {machine_id}")
        return

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Health Index', color=color)
    ax1.plot(df_total.index, df_total['health_index'], color=color, label='Health Index')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 1.05)

    # Secondary axis for Temperature
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Temperature (°C)', color=color)
    ax2.plot(df_total.index, df_total['temperature'], color=color, alpha=0.3, label='Temperature')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'Degradation and Thermal Profile for {machine_id}')
    fig.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    # Change this ID to one of your logs
    visualize_machine_health('M_0000')
