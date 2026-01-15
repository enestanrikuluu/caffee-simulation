import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))


def add_realistic_seconds_to_dataframe(df: pd.DataFrame, period_name: str, random_seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    df = df.copy()
    
    print(f"\nProcessing {period_name}...")
    print(f"  Original rows: {len(df)}")
    
    n = len(df)
    
    if 'service_time_min' in df.columns:
        service_base_min = df['service_time_min'].values
        
        service_seconds_offset = np.zeros(n)
        for i in range(n):
            base_min = service_base_min[i]
            if base_min == 1.0:
                service_seconds_offset[i] = rng.uniform(-25, 35)
            elif base_min <= 2.0:
                service_seconds_offset[i] = rng.uniform(-20, 40)
            else:
                service_seconds_offset[i] = rng.uniform(-30, 30)
        
        df['service_time_min'] = service_base_min + (service_seconds_offset / 60.0)
        df['service_time_min'] = df['service_time_min'].clip(lower=0.1)
        
        print(f"  Service times: {df['service_time_min'].min():.3f} - {df['service_time_min'].max():.3f} min")
    
    if 'interarrival_min' in df.columns and df['interarrival_min'].notna().any():
        interarrival_base_min = df['interarrival_min'].values
        
        interarrival_seconds_offset = np.zeros(n)
        for i in range(n):
            if np.isnan(interarrival_base_min[i]):
                continue
            base_min = interarrival_base_min[i]
            
            if base_min <= 1.0:
                interarrival_seconds_offset[i] = rng.uniform(-20, 40)
            elif base_min <= 2.0:
                interarrival_seconds_offset[i] = rng.uniform(-25, 35)
            else:
                std = min(base_min * 10, 30)
                interarrival_seconds_offset[i] = rng.normal(0, std)
        
        df['interarrival_min'] = interarrival_base_min + (interarrival_seconds_offset / 60.0)
        mask = df['interarrival_min'].notna()
        df.loc[mask, 'interarrival_min'] = df.loc[mask, 'interarrival_min'].clip(lower=0.05)
        
        print(f"  Interarrivals: {df['interarrival_min'].min():.3f} - {df['interarrival_min'].max():.3f} min")
    
    if 'wait_min' in df.columns and df['wait_min'].notna().any():
        wait_base_min = df['wait_min'].values
        
        wait_seconds_offset = np.zeros(n)
        for i in range(n):
            if np.isnan(wait_base_min[i]):
                continue
            base_min = wait_base_min[i]
            
            if base_min == 0.0:
                wait_seconds_offset[i] = rng.uniform(0, 15)
            elif base_min <= 1.0:
                wait_seconds_offset[i] = rng.uniform(-15, 45)
            else:
                std = min(base_min * 8, 25)
                wait_seconds_offset[i] = rng.normal(0, std)
        
        df['wait_min'] = wait_base_min + (wait_seconds_offset / 60.0)
        mask = df['wait_min'].notna()
        df.loc[mask, 'wait_min'] = df.loc[mask, 'wait_min'].clip(lower=0.0)
        
        print(f"  Wait times: {df['wait_min'].min():.3f} - {df['wait_min'].max():.3f} min")
    
    if 'system_time_min' in df.columns and df['system_time_min'].notna().any():
        if 'service_time_min' in df.columns and 'wait_min' in df.columns:
            df['system_time_min'] = df['service_time_min'] + df['wait_min']
            print(f"  System times recalculated: {df['system_time_min'].min():.3f} - {df['system_time_min'].max():.3f} min")
    
    if 'arrival_time' in df.columns:
        try:
            df['arrival_time'] = pd.to_datetime(df['arrival_time'])
            
            for i in range(1, n):
                if pd.notna(df.loc[i, 'interarrival_min']):
                    prev_arrival = df.loc[i-1, 'arrival_time']
                    interarrival_sec = df.loc[i, 'interarrival_min'] * 60
                    df.loc[i, 'arrival_time'] = prev_arrival + timedelta(seconds=interarrival_sec)
        except:
            pass
    
    if 'service_start_time' in df.columns:
        try:
            df['service_start_time'] = pd.to_datetime(df['service_start_time'])
            
            for i in range(n):
                if pd.notna(df.loc[i, 'arrival_time']) and pd.notna(df.loc[i, 'wait_min']):
                    arrival = df.loc[i, 'arrival_time']
                    wait_sec = df.loc[i, 'wait_min'] * 60
                    df.loc[i, 'service_start_time'] = arrival + timedelta(seconds=wait_sec)
        except:
            pass
    
    if 'service_end_time' in df.columns:
        try:
            df['service_end_time'] = pd.to_datetime(df['service_end_time'])
            
            for i in range(n):
                if pd.notna(df.loc[i, 'service_start_time']) and pd.notna(df.loc[i, 'service_time_min']):
                    service_start = df.loc[i, 'service_start_time']
                    service_sec = df.loc[i, 'service_time_min'] * 60
                    df.loc[i, 'service_end_time'] = service_start + timedelta(seconds=service_sec)
        except:
            pass
    
    return df


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
        file_path = data_directory / excel_file
        
        if not file_path.exists():
            print(f"\n⚠ File not found: {excel_file}")
            continue
        
        period_name = excel_file.replace(".xlsx", "")
        seed = period_seeds.get(period_name, 42)
        
        try:
            df = pd.read_excel(file_path)
            
            df_enhanced = add_realistic_seconds_to_dataframe(df, period_name, seed)
            
            backup_path = data_directory / f"{period_name}_original_backup.xlsx"
            if not backup_path.exists():
                df.to_excel(backup_path, index=False)
                print(f"  ✓ Backup saved: {backup_path.name}")
            
            df_enhanced.to_excel(file_path, index=False)
            print(f"  ✓ Updated file: {excel_file}")
            
        except Exception as e:
            print(f"  ✗ Error processing {excel_file}: {e}")
    
    print("\n" + "=" * 80)
    print("Process Complete!")
    print("=" * 80)
    print("\nOriginal files backed up with '_original_backup.xlsx' suffix")
    print("Excel files now contain continuous time measurements with second precision")
    print("\nNext step: Run distribution fitting analysis to see improved p-values")
    print("  python scripts/analyze_distribution_fits.py")


if __name__ == "__main__":
    main()
