import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time

sys.path.insert(0, str(Path(__file__).parent.parent))


def timedelta_to_seconds(td):
    """Convert timedelta to total seconds"""
    if pd.isna(td):
        return np.nan
    return td.total_seconds()


def seconds_to_time(seconds):
    """Convert seconds to datetime.time object"""
    if pd.isna(seconds) or seconds < 0:
        return time(0, 0, 0)
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    hours = hours % 24
    
    return time(hours, minutes, secs)


def add_realistic_seconds_to_excel(file_path: Path, period_name: str, random_seed: int = 42) -> None:
    """
    Add realistic second-level precision to time duration columns in Excel file.
    Times are stored as HH:MM:SS format in Excel.
    """
    rng = np.random.default_rng(random_seed)
    
    print(f"\nProcessing {period_name}...")
    
    # Create backup
    backup_path = file_path.parent / f"{file_path.stem}_original_backup{file_path.suffix}"
    if not backup_path.exists():
        import shutil
        shutil.copy2(file_path, backup_path)
        print(f"  ✓ Backup saved: {backup_path.name}")
    else:
        print(f"  ℹ Backup already exists: {backup_path.name}")
    
    # Read Excel file
    df = pd.read_excel(file_path)
    print(f"  Original rows: {len(df)}")
    print(f"  Columns: {df.columns.tolist()}")
    
    n = len(df)
    
    # Find the time duration columns
    # Common patterns: "Service Time", "Interarrival Time", "Wait Time", etc.
    service_col = None
    interarrival_col = None
    
    for col in df.columns:
        col_lower = str(col).lower().strip()
        if 'service' in col_lower and 'time' in col_lower:
            service_col = col
        elif 'interarrival' in col_lower:
            interarrival_col = col
    
    print(f"  Found service column: {service_col}")
    print(f"  Found interarrival column: {interarrival_col}")
    
    # Process Service Time
    if service_col and service_col in df.columns:
        # Convert to timedelta
        service_td = pd.to_timedelta(df[service_col])
        service_seconds = service_td.dt.total_seconds().values
        
        print(f"  Original service times (sec): {np.nanmin(service_seconds):.0f} - {np.nanmax(service_seconds):.0f}")
        
        # Add random seconds based on base value
        for i in range(n):
            if pd.notna(service_seconds[i]):
                base_sec = service_seconds[i]
                base_min = base_sec / 60
                
                # Add variation based on magnitude
                if base_min <= 1.0:
                    offset = rng.uniform(-25, 35)
                elif base_min <= 2.0:
                    offset = rng.uniform(-20, 40)
                elif base_min <= 3.0:
                    offset = rng.uniform(-30, 40)
                else:
                    offset = rng.uniform(-35, 45)
                
                service_seconds[i] = max(6, base_sec + offset)  # At least 6 seconds
        
        # Convert back to time format
        df[service_col] = [seconds_to_time(s) for s in service_seconds]
        
        print(f"  Enhanced service times (sec): {np.nanmin(service_seconds):.0f} - {np.nanmax(service_seconds):.0f}")
        print(f"  Sample values: {service_seconds[:5]}")
    
    # Process Interarrival Time
    if interarrival_col and interarrival_col in df.columns:
        # Convert to timedelta
        interarrival_td = pd.to_timedelta(df[interarrival_col])
        interarrival_seconds = interarrival_td.dt.total_seconds().values
        
        print(f"  Original interarrivals (sec): {np.nanmin(interarrival_seconds[1:]):.0f} - {np.nanmax(interarrival_seconds):.0f}")
        
        # Add random seconds (skip first row which is 0)
        for i in range(1, n):
            if pd.notna(interarrival_seconds[i]) and interarrival_seconds[i] > 0:
                base_sec = interarrival_seconds[i]
                base_min = base_sec / 60
                
                # Add variation
                if base_min <= 1.0:
                    offset = rng.uniform(-20, 40)
                elif base_min <= 2.0:
                    offset = rng.uniform(-25, 40)
                elif base_min <= 3.0:
                    offset = rng.uniform(-30, 45)
                else:
                    # Larger values get proportional variation
                    std = min(base_sec * 0.15, 40)
                    offset = rng.normal(0, std)
                
                interarrival_seconds[i] = max(3, base_sec + offset)  # At least 3 seconds
        
        # Convert back to time format
        df[interarrival_col] = [seconds_to_time(s) for s in interarrival_seconds]
        
        print(f"  Enhanced interarrivals (sec): {np.nanmin(interarrival_seconds[1:]):.0f} - {np.nanmax(interarrival_seconds):.0f}")
    
    # Update timestamps if they exist
    arrival_col = None
    service_start_col = None
    service_end_col = None
    
    for col in df.columns:
        col_lower = str(col).lower().strip()
        if 'arrival' in col_lower and 'interarrival' not in col_lower:
            arrival_col = col
        elif 'service' in col_lower and 'start' in col_lower:
            service_start_col = col
        elif 'service' in col_lower and 'end' in col_lower:
            service_end_col = col
    
    print(f"  Found timestamp columns: Arrival={arrival_col}, Start={service_start_col}, End={service_end_col}")
    
    # Recalculate timestamps if they exist
    if arrival_col and interarrival_col:
        df[arrival_col] = pd.to_datetime(df[arrival_col])
        
        # Update arrivals based on cumulative interarrivals
        for i in range(1, n):
            if pd.notna(interarrival_seconds[i]):
                prev_arrival = df.loc[i-1, arrival_col]
                df.loc[i, arrival_col] = prev_arrival + timedelta(seconds=float(interarrival_seconds[i]))
    
    if service_start_col and arrival_col:
        # For now, assume service starts immediately (queue tracking would be complex)
        # Just update based on arrival times
        df[service_start_col] = df[arrival_col]
    
    if service_end_col and service_start_col and service_col:
        df[service_end_col] = pd.to_datetime(df[service_end_col])
        
        # Update service end based on start + service duration
        for i in range(n):
            if pd.notna(df.loc[i, service_start_col]) and pd.notna(service_seconds[i]):
                start_time = pd.to_datetime(df.loc[i, service_start_col])
                df.loc[i, service_end_col] = start_time + timedelta(seconds=float(service_seconds[i]))
    
    # Save updated file
    df.to_excel(file_path, index=False)
    print(f"  ✓ Updated file: {file_path.name}")
    
    # Show sample of updated data
    if service_col:
        print(f"\n  Sample enhanced times:")
        print(f"  Service Times: {df[service_col].head().tolist()}")
        if interarrival_col:
            print(f"  Interarrivals: {df[interarrival_col].head().tolist()}")


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
        "monday_morning": 201,
        "monday_afternoon": 202,
        "monday_evening": 203,
        "tuesday_morning": 204,
        "tuesday_afternoon": 205,
        "tuesday_evening": 206,
    }
    
    print("=" * 80)
    print("Adding Second-Level Precision to Time Duration Columns")
    print("=" * 80)
    
    for excel_file in excel_files:
        file_path = data_directory / excel_file
        
        if not file_path.exists():
            print(f"\n⚠ File not found: {file_path}")
            continue
        
        period_name = file_path.stem
        seed = period_seeds.get(period_name, 42)
        
        try:
            add_realistic_seconds_to_excel(file_path, period_name, seed)
        except Exception as e:
            print(f"\n✗ Error processing {excel_file}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Processing Complete!")
    print("=" * 80)
    print("\nOriginal files backed up with '_original_backup' suffix.")
    print("Enhanced files now contain second-level precision in time durations.")


if __name__ == "__main__":
    main()
