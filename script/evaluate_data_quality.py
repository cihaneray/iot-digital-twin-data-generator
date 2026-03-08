import glob
import os

import pandas as pd


def load_telemetry(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "telemetry", "*.parquet")))
    if not files:
        print("No telemetry data found in data/telemetry/")
        return None

    print(f"Found {len(files)} telemetry files. Loading sample...")
    if len(files) > 6:
        mid = len(files) // 2
        files_to_load = files[:2] + files[mid:mid + 2] + files[-2:]
    else:
        files_to_load = files

    dfs = []
    for f in files_to_load:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not dfs:
        return None

    combined_df = pd.concat(dfs)
    print(f"Loaded telemetry shape: {combined_df.shape}")
    return combined_df


def load_generated_logs(data_dir):
    log_path = os.path.join(data_dir, "logs", "logs.parquet")
    if not os.path.exists(log_path):
        print("No generated logs found in data/logs/logs.parquet")
        return None
    return pd.read_parquet(log_path)


def load_example_logs(script_dir):
    example_path = os.path.join(script_dir, "example.csv")
    if not os.path.exists(example_path):
        print(f"No example logs found at {example_path}")
        return None
    try:
        return pd.read_csv(filepath_or_buffer=example_path)
    except Exception as e:
        print(f"Error reading example logs: {e}")
        return None


def load_real_telemetry(script_dir):
    data_path = os.path.join(script_dir, "data.csv")
    if not os.path.exists(data_path):
        print(f"No real telemetry data found at {data_path}")
        return None

    print(f"Loading real telemetry from {data_path}...")
    try:
        df = pd.read_csv(data_path)
        print(f"Loaded Real Data Shape: {df.shape}")

        rename_map = {
            'volt': 'voltage',
            'rotate': 'speed',
            'vibration': 'vibration_intensity'
        }

        available_cols = [c for c in rename_map.keys() if c in df.columns]
        if 'pressure' in df.columns: available_cols.append('pressure')

        print(f"Found real columns: {available_cols}")

        df.rename(columns=rename_map, inplace=True)

        return df
    except Exception as e:
        print(f"Error reading real telemetry: {e}")
        return None


def check_physics(telemetry_df):
    print("\n" + "=" * 40)
    print("INTERNAL PHYSICS CONSISTENCY CHECK")
    print("=" * 40)

    running = telemetry_df[telemetry_df['speed'] > 100].copy()

    if running.empty:
        print("No running state data found (speed > 100). Cannot perform physics checks.")
        return

    print(f"Analyzing {len(running)} data points where machine is running...")

    running['temp_diff'] = running['temperature'] - running['ambient_temperature']
    corr_temp_speed = running['temp_diff'].corr(running['speed'])
    print(f"1. Correlation (Temp Rise vs Speed): {corr_temp_speed:.2f}")
    if corr_temp_speed > 0.5:
        print("   [INFO: PASS] Strong positive correlation (Speed generates heat).")
    elif corr_temp_speed > 0:
        print("   [INFO: PASS] Positive correlation detected.")
    else:
        print("   [INFO: WARNING] Negative or no correlation. Check thermal model.")

    corr_vib_health = running['vibration_intensity'].corr(running['health_index'])
    print(f"2. Correlation (Vibration vs Health Index): {corr_vib_health:.2f}")
    if corr_vib_health < -0.5:
        print("   [INFO: PASS] Strong negative correlation (Health drops -> Vibration spikes).")
    elif corr_vib_health < 0:
        print("   [INFO: PASS (WEAK)] Weak negative correlation.")
    else:
        print("   [ERROR: FAIL] Positive correlation? Better health shouldn't mean more vibration.")

    corr_volt_load = running['voltage'].corr(running['load'])
    print(f"3. Correlation (Voltage vs Load): {corr_volt_load:.2f}")
    if corr_volt_load < 0:
        print("   [INFO: PASS] Voltage sags with Load.")
    else:
        print("   Note: Voltage might be regulated or random noise dominates load sag.")

    p_min, p_max = running['pressure'].min(), running['pressure'].max()
    print(f"4. Pressure Range: {p_min:.2f} - {p_max:.2f}")
    if p_min > 0:
        print("   [INFO: PASS] Pressure stays positive.")
    else:
        print("   [ERROR: FAIL] Negative pressure detected.")


def compare_logs(gen_logs, real_logs):
    print("EXTERNAL VALIDATION (LOGS)")

    if gen_logs is None or real_logs is None:
        print("Skipping comparison due to missing data.")
        return

    gen_crit = gen_logs[gen_logs['level'] == 'CRITICAL'].copy()

    real_crit = real_logs[real_logs['level'] == 'CRITICAL'].copy()

    def clean_code(c):
        if ":" in str(c):
            return str(c).split(":")[-1].strip()
        return str(c)

    gen_crit['type'] = gen_crit['code'].apply(clean_code)
    real_crit['type'] = real_crit['code'].apply(clean_code)

    print("--- Failure Type Distribution (Generated) ---")
    gen_dist = gen_crit['type'].value_counts(normalize=True)
    print(gen_dist)

    print("\n--- Failure Type Distribution (Real/Example) ---")
    real_dist = real_crit['type'].value_counts(normalize=True)
    print(real_dist)

    print("\n--- Reality Score ---")
    gen_top = set(gen_dist.index)
    real_top = set(real_dist.index)

    common = gen_top.intersection(real_top)
    print(f"Common Failure Modes: {common}")

    if len(common) > 0:
        print(
            f"[INFO: PASS] The generated data successfully reproduces these real-world failure modes: {', '.join(common)}")
    else:
        print("[ERROR: FAIL] No overlap in failure modes found. Check naming conventions or generation config.")

    def get_ratio(df):
        counts = df['level'].value_counts()
        w = counts.get('WARNING', 0)
        c = counts.get('CRITICAL', 0)
        if c == 0: return 0
        return w / c

    gen_ratio = get_ratio(gen_logs)
    real_ratio = get_ratio(real_logs)

    print(f"\nWarning/Critical Ratio - Generated: {gen_ratio:.2f}")
    print(f"Warning/Critical Ratio - Real:      {real_ratio:.2f}")

    if abs(gen_ratio - real_ratio) < 5:
        print("[INFO: PASS] Alerting sensitivity is similar to real data.")
    else:
        print("[INFO: WARNING] Alerting sensitivity differs significantly.")


def compare_telemetry_distributions(synthetic_df, real_df):
    print("EXTERNAL VALIDATION (TELEMETRY STATISTICS)")

    if synthetic_df is None or real_df is None:
        print("Skipping telemetry comparison (missing data).")
        return

    cols = ['voltage', 'speed', 'pressure', 'vibration_intensity']

    synth_active = synthetic_df[synthetic_df['speed'] > 10]

    if 'speed' in real_df.columns:
        real_active = real_df[real_df['speed'] > 10]
        if len(real_active) == 0:
            print("Warning: Real data has no speed > 10? Using all real data.")
            real_active = real_df
    else:
        real_active = real_df

    stats = []

    for col in cols:
        if col not in synthetic_df.columns or col not in real_df.columns:
            continue

        s_dat = synth_active[col]
        r_dat = real_active[col]

        stats.append({
            "Metric": f"{col} (Mean)",
            "Synthetic": f"{s_dat.mean():.2f}",
            "Real": f"{r_dat.mean():.2f}",
            "Diff %": f"{abs(s_dat.mean() - r_dat.mean()) / (r_dat.mean() + 1e-6) * 100:.1f}%"
        })
        stats.append({
            "Metric": f"{col} (StdDev)",
            "Synthetic": f"{s_dat.std():.2f}",
            "Real": f"{r_dat.std():.2f}",
            "Diff %": f"{abs(s_dat.std() - r_dat.std()) / (r_dat.std() + 1e-6) * 100:.1f}%"
        })
        stats.append({
            "Metric": f"{col} (Range)",
            "Synthetic": f"[{s_dat.min():.1f}, {s_dat.max():.1f}]",
            "Real": f"[{r_dat.min():.1f}, {r_dat.max():.1f}]",
            "Diff %": "-"
        })
        stats.append({"Metric": "-" * 20, "Synthetic": "", "Real": "", "Diff %": ""})

    df_stats = pd.DataFrame(stats)
    print(df_stats.to_string(index=False))


def main():
    current_file = os.path.abspath(__file__)
    script_dir = os.path.dirname(current_file)
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, "data")

    print(f"Working with Root: {project_root}")

    tel_df = load_telemetry(data_dir)
    gen_logs = load_generated_logs(data_dir)
    real_logs = load_example_logs(script_dir)
    real_telemetry = load_real_telemetry(script_dir)

    if tel_df is not None:
        check_physics(tel_df)
    else:
        print("Skipping physics check (no telemetry).")

    compare_telemetry_distributions(tel_df, real_telemetry)
    compare_logs(gen_logs, real_logs)


if __name__ == "__main__":
    main()
