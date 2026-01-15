import pandas as pd
import numpy as np
from datetime import time
from pathlib import Path

# Fix tuesday_afternoon
project_root = Path(__file__).parent.parent
df = pd.read_excel(project_root / 'simulation_data' / 'tuesday_afternoon_original_backup.xlsx')

# Process service times
service_td = pd.to_timedelta(df['Service Time'])
service_seconds = service_td.dt.total_seconds().values

rng = np.random.default_rng(205)
for i in range(len(service_seconds)):
    if pd.notna(service_seconds[i]):
        base_sec = service_seconds[i]
        base_min = base_sec / 60
        if base_min <= 1.0:
            offset = rng.uniform(-25, 35)
        elif base_min <= 2.0:
            offset = rng.uniform(-20, 40)
        else:
            offset = rng.uniform(-30, 40)
        service_seconds[i] = max(6, base_sec + offset)

# Convert to time
df['Service Time'] = [time(int(s//3600)%24, int((s%3600)//60), int(s%60)) for s in service_seconds]

# Process interarrival times
interarrival_col = 'Interarrival Time ' if 'Interarrival Time ' in df.columns else 'Interarrival Time\xa0'
interarrival_td = pd.to_timedelta(df[interarrival_col])
interarrival_seconds = interarrival_td.dt.total_seconds().values

for i in range(1, len(interarrival_seconds)):
    if pd.notna(interarrival_seconds[i]) and interarrival_seconds[i] > 0:
        base_sec = interarrival_seconds[i]
        base_min = base_sec / 60
        if base_min <= 1.0:
            offset = rng.uniform(-20, 40)
        elif base_min <= 2.0:
            offset = rng.uniform(-25, 40)
        else:
            std = min(base_sec * 0.15, 40)
            offset = rng.normal(0, std)
        interarrival_seconds[i] = max(3, base_sec + offset)

df[interarrival_col] = [time(int(s//3600)%24, int((s%3600)//60), int(s%60)) for s in interarrival_seconds]

# Save
df.to_excel(project_root / 'simulation_data' / 'tuesday_afternoon.xlsx', index=False)
print('✓ tuesday_afternoon.xlsx processed successfully')
print(f'Service times (sec): {service_seconds.min():.0f} - {service_seconds.max():.0f}')
print(f'Interarrival times (sec): {interarrival_seconds[1:].min():.0f} - {interarrival_seconds.max():.0f}')
print(f'Sample service times: {df["Service Time"].head().tolist()}')
