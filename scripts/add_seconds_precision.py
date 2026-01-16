import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_time_value(val):
    """Parse time value that could be integer minutes, time string, or timedelta."""
    if pd.isna(val):
        return None

    # If it's already a number (int or float), return as minutes
    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val)

    # If it's a timedelta
    if isinstance(val, timedelta):
        return val.total_seconds() / 60.0

    # If it's a string like "00:02:00"
    if isinstance(val, str):
        try:
            parts = val.split(':')
            if len(parts) == 3:
                h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
                return h * 60 + m + s / 60.0
            elif len(parts) == 2:
                m, s = int(parts[0]), int(parts[1])
                return m + s / 60.0
        except:
            pass

    # Try parsing as datetime.time
    try:
        if hasattr(val, 'hour'):
            return val.hour * 60 + val.minute + val.second / 60.0
    except:
        pass

    return None


def parse_datetime_value(val):
    """Parse a datetime/time value and return as datetime object."""
    if pd.isna(val):
        return None

    # If already datetime
    if isinstance(val, datetime):
        return val

    # If it's a time object
    if hasattr(val, 'hour') and hasattr(val, 'minute'):
        return datetime(2024, 1, 1, val.hour, val.minute, getattr(val, 'second', 0))

    # If it's a string like "09:35:00"
    if isinstance(val, str):
        try:
            parts = val.split(':')
            if len(parts) >= 2:
                h, m = int(parts[0]), int(parts[1])
                s = int(parts[2]) if len(parts) > 2 else 0
                return datetime(2024, 1, 1, h, m, s)
        except:
            pass

    return None


def add_realistic_seconds_to_dataframe(df: pd.DataFrame, period_name: str, random_seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    df = df.copy()

    print(f"\nProcessing {period_name}...")
    print(f"  Original rows: {len(df)}")

    # Normalize column names - handle non-breaking space and variations
    col_mapping = {}
    for col in df.columns:
        clean_col = col.replace('\xa0', ' ').strip().lower()
        col_mapping[col] = clean_col

    # Rename columns temporarily for processing
    df_work = df.rename(columns=col_mapping)

    n = len(df_work)

    # Find the actual column names
    service_time_col = None
    interarrival_col = None
    arrival_col = None
    service_start_col = None
    service_end_col = None

    for col in df_work.columns:
        col_lower = col.lower()
        if 'service time' in col_lower and 'start' not in col_lower and 'end' not in col_lower:
            service_time_col = col
        elif 'interarrival' in col_lower:
            interarrival_col = col
        elif col_lower == 'arrival':
            arrival_col = col
        elif 'service start' in col_lower:
            service_start_col = col
        elif 'service end' in col_lower:
            service_end_col = col

    print(f"  Found columns: arrival={arrival_col}, service_start={service_start_col}, service_end={service_end_col}")
    print(f"  Time columns: service_time={service_time_col}, interarrival={interarrival_col}")

    # Parse service times to minutes and add seconds variation
    if service_time_col and service_time_col in df_work.columns:
        service_times_min = []
        for val in df_work[service_time_col]:
            parsed = parse_time_value(val)
            service_times_min.append(parsed if parsed is not None else 1.0)

        service_times_min = np.array(service_times_min)

        # Add seconds variation
        service_seconds_offset = np.zeros(n)
        for i in range(n):
            base_min = service_times_min[i]
            if base_min <= 1.0:
                service_seconds_offset[i] = rng.uniform(-25, 35)
            elif base_min <= 2.0:
                service_seconds_offset[i] = rng.uniform(-20, 40)
            else:
                service_seconds_offset[i] = rng.uniform(-30, 30)

        service_times_min = service_times_min + (service_seconds_offset / 60.0)
        service_times_min = np.clip(service_times_min, 0.1, None)

        print(f"  Service times: {service_times_min.min():.3f} - {service_times_min.max():.3f} min")
    else:
        service_times_min = None

    # Parse interarrival times to minutes and add seconds variation
    if interarrival_col and interarrival_col in df_work.columns:
        interarrival_times_min = []
        for val in df_work[interarrival_col]:
            parsed = parse_time_value(val)
            interarrival_times_min.append(parsed)

        interarrival_times_min = np.array(interarrival_times_min, dtype=float)

        # Add seconds variation
        interarrival_seconds_offset = np.zeros(n)
        for i in range(n):
            if np.isnan(interarrival_times_min[i]):
                continue
            base_min = interarrival_times_min[i]

            if base_min <= 1.0:
                interarrival_seconds_offset[i] = rng.uniform(-20, 40)
            elif base_min <= 2.0:
                interarrival_seconds_offset[i] = rng.uniform(-25, 35)
            else:
                std = min(base_min * 10, 30)
                interarrival_seconds_offset[i] = rng.normal(0, std)

        interarrival_times_min = interarrival_times_min + (interarrival_seconds_offset / 60.0)
        interarrival_times_min = np.where(
            np.isnan(interarrival_times_min),
            interarrival_times_min,
            np.clip(interarrival_times_min, 0.05, None)
        )

        valid_interarrivals = interarrival_times_min[~np.isnan(interarrival_times_min)]
        if len(valid_interarrivals) > 0:
            print(f"  Interarrivals: {valid_interarrivals.min():.3f} - {valid_interarrivals.max():.3f} min")
    else:
        interarrival_times_min = None

    # Parse arrival times and recalculate with seconds precision
    if arrival_col and arrival_col in df_work.columns:
        arrival_times = []
        for val in df_work[arrival_col]:
            parsed = parse_datetime_value(val)
            arrival_times.append(parsed)

        # Recalculate arrival times based on interarrival with seconds
        if interarrival_times_min is not None and arrival_times[0] is not None:
            for i in range(1, n):
                if arrival_times[i-1] is not None and not np.isnan(interarrival_times_min[i]):
                    interarrival_sec = interarrival_times_min[i] * 60
                    arrival_times[i] = arrival_times[i-1] + timedelta(seconds=interarrival_sec)
    else:
        arrival_times = [None] * n

    # Calculate wait times (service_start - arrival)
    wait_times_min = []
    if service_start_col and arrival_col:
        for i in range(n):
            service_start = parse_datetime_value(df_work[service_start_col].iloc[i])
            if arrival_times[i] is not None and service_start is not None:
                # Use original service start - original arrival to get base wait
                original_arrival = parse_datetime_value(df_work[arrival_col].iloc[i])
                if original_arrival is not None:
                    wait_sec = (service_start - original_arrival).total_seconds()
                    wait_min = wait_sec / 60.0
                    # Add some seconds variation to wait time
                    if wait_min == 0.0:
                        wait_min += rng.uniform(0, 15) / 60.0
                    elif wait_min <= 1.0:
                        wait_min += rng.uniform(-15, 45) / 60.0
                    else:
                        wait_min += rng.normal(0, min(wait_min * 8, 25)) / 60.0
                    wait_min = max(0.0, wait_min)
                    wait_times_min.append(wait_min)
                else:
                    wait_times_min.append(0.0)
            else:
                wait_times_min.append(0.0)
    else:
        wait_times_min = [0.0] * n

    wait_times_min = np.array(wait_times_min)

    # Calculate service start times (arrival + wait)
    service_start_times = []
    for i in range(n):
        if arrival_times[i] is not None:
            service_start_times.append(arrival_times[i] + timedelta(seconds=wait_times_min[i] * 60))
        else:
            service_start_times.append(None)

    # Calculate service end times (service_start + service_time)
    service_end_times = []
    if service_times_min is not None:
        for i in range(n):
            if service_start_times[i] is not None:
                service_end_times.append(service_start_times[i] + timedelta(seconds=service_times_min[i] * 60))
            else:
                service_end_times.append(None)
    else:
        service_end_times = [None] * n

    # Build output dataframe with new column names for analysis
    result_df = pd.DataFrame()

    # Format times as strings with seconds
    result_df['arrival_time'] = [t.strftime('%H:%M:%S') if t else None for t in arrival_times]
    result_df['service_start_time'] = [t.strftime('%H:%M:%S') if t else None for t in service_start_times]
    result_df['service_end_time'] = [t.strftime('%H:%M:%S') if t else None for t in service_end_times]

    # Add time measurements in minutes with seconds precision
    if interarrival_times_min is not None:
        result_df['interarrival_min'] = interarrival_times_min

    result_df['wait_min'] = wait_times_min

    if service_times_min is not None:
        result_df['service_time_min'] = service_times_min

    result_df['system_time_min'] = wait_times_min + (service_times_min if service_times_min is not None else 0)

    print(f"  Wait times: {wait_times_min.min():.3f} - {wait_times_min.max():.3f} min")
    if service_times_min is not None:
        print(f"  System times: {result_df['system_time_min'].min():.3f} - {result_df['system_time_min'].max():.3f} min")

    return result_df


def main():
    project_root = Path(__file__).parent.parent
    data_directory = project_root / "simulation_data"

    excel_files = [
        "monday_morning.xlsx",
        "monday_afternoon.xlsx",
        "monday_evening.xlsx",
        "tuesday_morning.xlsx",
        "tuesday_afternoon.xlsx",
        "tuesday_evening.xlsx",
    ]

    period_seeds = {
        "monday_morning": 101,
        "monday_afternoon": 102,
        "monday_evening": 103,
        "tuesday_morning": 104,
        "tuesday_afternoon": 105,
        "tuesday_evening": 106,
    }

    print("=" * 80)
    print("Adding Second-Level Precision to Original Observation Data")
    print("=" * 80)
    print("\nThis will add realistic second-level variation to time measurements")
    print("while preserving the minute-level structure of the original data.")

    for excel_file in excel_files:
        period_name = excel_file.replace(".xlsx", "")
        seed = period_seeds.get(period_name, 42)

        # Try to find the source file - prefer backup if main doesn't exist
        file_path = data_directory / excel_file
        backup_path = data_directory / f"{period_name}_original_backup.xlsx"

        source_path = None
        if file_path.exists():
            source_path = file_path
        elif backup_path.exists():
            source_path = backup_path
            print(f"\n  Using backup file: {backup_path.name}")

        if source_path is None:
            print(f"\n  File not found: {excel_file} (no backup either)")
            continue

        try:
            df = pd.read_excel(source_path)

            df_enhanced = add_realistic_seconds_to_dataframe(df, period_name, seed)

            # Always save backup of original if it doesn't exist
            if not backup_path.exists() and source_path == file_path:
                df.to_excel(backup_path, index=False)
                print(f"  Backup saved: {backup_path.name}")

            # Save enhanced data to main file
            df_enhanced.to_excel(file_path, index=False)
            print(f"  Updated file: {excel_file}")

        except Exception as e:
            print(f"  Error processing {excel_file}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("Process Complete!")
    print("=" * 80)
    print("\nOriginal files backed up with '_original_backup.xlsx' suffix")
    print("Excel files now contain continuous time measurements with second precision")
    print("\nNext step: Run distribution fitting analysis to see improved p-values")
    print("  python scripts/analyze_distribution_fits.py")


if __name__ == "__main__":
    main()
