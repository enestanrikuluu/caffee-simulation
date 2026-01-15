import pandas as pd
from pathlib import Path
from typing import Dict
from cafe_sim.period_key import PeriodKey


class DataLoader:
    def __init__(self, data_directory: Path):
        self._data_directory = data_directory
        self._file_mapping = {
            PeriodKey.MONDAY_MORNING: "monday_morning.xlsx",
            PeriodKey.MONDAY_AFTERNOON: "monday_afternoon.xlsx",
            PeriodKey.MONDAY_EVENING: "monday_evening.xlsx",
            PeriodKey.TUESDAY_MORNING: "tuesday_morning.xlsx",
            PeriodKey.TUESDAY_AFTERNOON: "tuesday_afternoon.xlsx",
            PeriodKey.TUESDAY_EVENING: "tuesday_evening.xlsx",
        }

    def load_all_periods(self) -> Dict[PeriodKey, pd.DataFrame]:
        return {
            period_key: self._load_period(period_key)
            for period_key in PeriodKey
        }

    def _load_period(self, period_key: PeriodKey) -> pd.DataFrame:
        file_name = self._file_mapping[period_key]
        file_path = self._data_directory / file_name
        raw_df = self._load_excel(file_path)
        return self._normalize_columns(raw_df)

    def _load_excel(self, path: Path) -> pd.DataFrame:
        return pd.read_excel(path, engine="openpyxl")

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
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
