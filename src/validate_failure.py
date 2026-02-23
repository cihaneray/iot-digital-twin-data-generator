import pandas as pd


def validate_failure_lead_times(log_path='data/logs/logs.parquet'):
    try:
        # Load logs and ensure index is datetime
        df = pd.read_parquet(log_path)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Identify unique machines that have both a warning and a failure
        machines = df['machineID'].unique()
        lead_times = []

        print(f"{'Machine ID':<12} | {'Fault Type':<15} | {'Lead Time (Hours)':<18}")
        print("-" * 50)

        for mid in machines:
            m_logs = df[df['machineID'] == mid]

            # Find occurrences of warnings and failures
            warnings = m_logs[m_logs['level'] == 'WARNING']
            failures = m_logs[m_logs['level'] == 'CRITICAL']

            # Match each warning with the very next failure for that machine
            for _, warn_row in warnings.iterrows():
                warn_time = warn_row.name
                # Get the first failure that happens AFTER this warning
                subsequent_failures = failures[failures.index > warn_time]

                if not subsequent_failures.empty:
                    fail_time = subsequent_failures.index[0]
                    diff = fail_time - warn_time
                    hours = diff.total_seconds() / 3600

                    fault_name = warn_row['code'].replace("Anomaly Detected: ", "")
                    lead_times.append(hours)

                    print(f"{mid:<12} | {fault_name:<15} | {hours:<18.2f}")

        if lead_times:
            avg_lead = sum(lead_times) / len(lead_times)
            print("-" * 50)
            print(f"Average Lead Time: {avg_lead:.2f} hours")
        else:
            print("No complete Anomaly-to-Failure sequences found.")

    except FileNotFoundError:
        print(f"Error: Could not find log file at {log_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    validate_failure_lead_times()
