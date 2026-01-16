import pandas as pd
from pathlib import Path
from cafe_sim.period_key import (
    MONDAY_MORNING, MONDAY_AFTERNOON, MONDAY_EVENING,
    TUESDAY_MORNING, TUESDAY_AFTERNOON, TUESDAY_EVENING,
    ALL_PERIOD_KEYS
)


FILE_MAPPING = {
    MONDAY_MORNING: "monday_morning.xlsx",
    MONDAY_AFTERNOON: "monday_afternoon.xlsx",
    MONDAY_EVENING: "monday_evening.xlsx",
    TUESDAY_MORNING: "tuesday_morning.xlsx",
    TUESDAY_AFTERNOON: "tuesday_afternoon.xlsx",
    TUESDAY_EVENING: "tuesday_evening.xlsx",
}


def load_all_periods(data_directory):
    return {
        period_key: load_period(data_directory, period_key)
        for period_key in ALL_PERIOD_KEYS
    }


def load_period(data_directory, period_key):
    file_name = FILE_MAPPING[period_key]
    file_path = Path(data_directory) / file_name
    raw_df = pd.read_excel(file_path, engine="openpyxl")
    return normalize_columns(raw_df)


def normalize_columns(df):
    normalized = df.copy()

    normalized = normalized.loc[:, ~normalized.columns.str.contains('^Unnamed')]

    column_mapping = {}
    for col in normalized.columns:
        col_lower = str(col).lower().strip().replace("\xa0", "").replace(" ", "")

        if "arrival" == col_lower or col_lower == "arrivaltime":
            column_mapping[col] = "arrival_time"
        elif "servicestart" in col_lower:
            column_mapping[col] = "service_start_time"
        elif "serviceend" in col_lower:
            column_mapping[col] = "service_end_time"
        elif "interarrival" in col_lower:
            column_mapping[col] = "interarrival_min"
        elif "servicetime" in col_lower or "serviceduration" in col_lower:
            column_mapping[col] = "service_time_min"

    if column_mapping:
        normalized = normalized.rename(columns=column_mapping)

    return normalized
