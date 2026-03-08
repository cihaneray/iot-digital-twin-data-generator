import glob
import os

import pandas as pd


def load_telemetry(data_dir):
    print("Searching for telemetry files...")
    files = sorted(glob.glob(os.path.join(data_dir, "telemetry", "*.parquet")))
    if not files:
        print("No telemetry data found in data/telemetry/")
        return None

    print(f"Found {len(files)} files. Loading full history for analysis...")

    dfs = []
    req_cols = ['timestamp', 'machineID', 'health_index', 'rul_seconds', 'fault_active']

    for f in files:
        try:
            try:
                df = pd.read_parquet(f, columns=req_cols)
            except Exception:
                df = pd.read_parquet(f)
                missing = [c for c in req_cols if c not in df.columns]
                if missing:
                    continue

            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")

            if not dfs:
                return None

    full_df = pd.concat(dfs)

    if 'timestamp' not in full_df.columns and isinstance(full_df.index, pd.DatetimeIndex):
        full_df.reset_index(inplace=True)
        if 'timestamp' not in full_df.columns:
            full_df.rename(columns={'index': 'timestamp'}, inplace=True)

    if 'timestamp' in full_df.columns and not pd.api.types.is_datetime64_any_dtype(full_df['timestamp']):
        full_df['timestamp'] = pd.to_datetime(full_df['timestamp'])

    if 'timestamp' in full_df.columns:
        full_df.sort_values('timestamp', inplace=True)
    else:
        print("Error: Could not find 'timestamp' column even after reset_index.")
        return None

    return full_df


def evaluate_machine_health(df, machine_id):
    print(f"\n{'=' * 20}\nMachine: {machine_id}\n{'=' * 20}")

    m_df = df[df['machineID'] == machine_id].copy()
    if m_df.empty:
        print("No data.")
        return

    if 'rul_seconds' not in m_df.columns:
        print("Column 'rul_seconds' not found.")
        return

    rul_df = m_df.dropna(subset=['rul_seconds'])

    if rul_df.empty:
        print("No degradation segments found (RUL is always Null/None). Machine might be healthy entire time.")
        return

    print(f"Found {len(rul_df)} data points with active RUL estimates.")

    rul_df = rul_df.sort_values('timestamp')
    rul_df['rul_change'] = rul_df['rul_seconds'].diff()

    cycle_starts = rul_df.index[rul_df['rul_change'] > 600].tolist()

    if not cycle_starts or cycle_starts[0] != rul_df.index[0]:
        cycle_starts.insert(0, rul_df.index[0])

    print(f"Detected {len(cycle_starts)} degradation cycle(s).")

    for i, start_idx in enumerate(cycle_starts):
        start_pos = rul_df.index.get_loc(start_idx)
        if i < len(cycle_starts) - 1:
            end_pos = rul_df.index.get_loc(cycle_starts[i + 1])
        else:
            end_pos = len(rul_df)

        cycle_data = rul_df.iloc[start_pos:end_pos]

        if len(cycle_data) < 10:
            continue

        analyze_cycle(cycle_data, i + 1)


def analyze_cycle(cycle_df, cycle_num):
    print(f"\n--- Cycle {cycle_num} Analysis ---")

    start_time = cycle_df['timestamp'].iloc[0]
    end_time = cycle_df['timestamp'].iloc[-1]
    duration = end_time - start_time

    print(f"Time Range: {start_time} to {end_time} (Duration: {duration})")
    print(f"RUL Range: {cycle_df['rul_seconds'].max() / 60:.1f} min -> {cycle_df['rul_seconds'].min() / 60:.1f} min")
    print(f"Health Range: {cycle_df['health_index'].max():.4f} -> {cycle_df['health_index'].min():.4f}")

    t_seconds = (cycle_df['timestamp'] - start_time).dt.total_seconds()
    rul = cycle_df['rul_seconds']

    corr = t_seconds.corr(rul)
    print(f"Correlation (Time vs RUL): {corr:.5f} (Target: -1.0)")

    if corr > -0.9:
        print("[[Warning]]: RUL does not decrease linearly with time. Prediction might be unstable.")
    else:
        print("[[Ok]]: RUL decreases linearly with time.")

    predicted_failures = cycle_df['timestamp'] + pd.to_timedelta(cycle_df['rul_seconds'], unit='s')

    mean_pred_time = predicted_failures.mean()
    std_pred_seconds = (predicted_failures - mean_pred_time).dt.total_seconds().std()

    print(f"Prediction Consistency (StdDev): {std_pred_seconds:.2f} seconds")
    if std_pred_seconds < 60:
        print("[[Ok]]: RUL prediction is highly consistent (stable target).")
    else:
        print("[[Warning]]: RUL prediction varies significantly during the cycle.")

    h_diff = cycle_df['health_index'].diff().fillna(-0.1)
    increases = (h_diff > 0.01).sum()
    if increases == 0:
        print("[[Ok]]: Health Index is strictly degrading (or flat).")
    else:
        print(f"[[Warning]]: Health Index increased {increases} times significantly. Physics violation?")


def main():
    current_file = os.path.abspath(__file__)
    script_dir = os.path.dirname(current_file)
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, "data")

    df = load_telemetry(data_dir)
    if df is None:
        return

    top_machines = df['machineID'].value_counts().head(3).index.tolist()

    for m_id in top_machines:
        evaluate_machine_health(df, m_id)


if __name__ == "__main__":
    main()
