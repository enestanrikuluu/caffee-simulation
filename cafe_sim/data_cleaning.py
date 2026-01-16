import pandas as pd
import numpy as np
import datetime


def clean_data(df, period_key):
    cleaned = df.copy()
    cleaned = coerce_time_columns(cleaned)
    cleaned = drop_invalid_rows(cleaned)
    cleaned = compute_derived_metrics(cleaned)
    cleaned = add_period_metadata(cleaned, period_key)
    return cleaned


def drop_invalid_rows(df):
    result = df.copy()

    result = result.dropna(how="all")

    if "service_time_min" in result.columns:
        valid_service = (
            pd.to_numeric(result["service_time_min"], errors="coerce") > 0
        )
        result = result[valid_service]

    result = result.reset_index(drop=True)
    return result


def coerce_time_columns(df):
    result = df.copy()

    time_columns = [
        "interarrival_min",
        "service_time_min",
        "arrival_time",
        "service_start_time",
        "service_end_time"
    ]

    for col in time_columns:
        if col in result.columns:
            result[col] = coerce_minutes(result[col])

    return result


def coerce_minutes(series):
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.dt.hour * 60 + series.dt.minute + series.dt.second / 60.0

    if pd.api.types.is_timedelta64_dtype(series):
        return series.dt.total_seconds() / 60.0

    numeric_series = pd.to_numeric(series, errors="coerce")

    if series.dtype == object:
        converted = series.copy()
        for idx, val in series.items():
            if isinstance(val, datetime.time):
                converted.loc[idx] = val.hour * 60 + val.minute + val.second / 60.0

        numeric_series = pd.to_numeric(converted, errors="coerce")

        if numeric_series.isna().any():
            timedelta_series = pd.to_timedelta(converted, errors="coerce")
            if timedelta_series.notna().any():
                time_mask = timedelta_series.notna() & numeric_series.isna()
                if time_mask.any():
                    numeric_series.loc[time_mask] = timedelta_series[time_mask].dt.total_seconds() / 60.0

        if numeric_series.isna().any():
            potential_time = pd.to_datetime(converted, errors="coerce")
            if potential_time.notna().any():
                time_mask = potential_time.notna() & numeric_series.isna()
                if time_mask.any():
                    hours = potential_time[time_mask].dt.hour
                    minutes = potential_time[time_mask].dt.minute
                    seconds = potential_time[time_mask].dt.second
                    numeric_series.loc[time_mask] = hours * 60 + minutes + seconds / 60.0

    return numeric_series


def compute_derived_metrics(df):
    result = df.copy()

    has_timestamps = all(
        col in result.columns
        for col in ["arrival_time", "service_start_time", "service_end_time"]
    )

    if has_timestamps:
        if "service_time_min" not in result.columns or result["service_time_min"].isna().any():
            result["service_time_min"] = (
                result["service_end_time"] - result["service_start_time"]
            )

        if "wait_min" not in result.columns:
            result["wait_min"] = (
                result["service_start_time"] - result["arrival_time"]
            )
            result["wait_min"] = result["wait_min"].clip(lower=0)

        if "system_time_min" not in result.columns:
            result["system_time_min"] = (
                result["service_end_time"] - result["arrival_time"]
            )

    return result


def add_period_metadata(df, period_key):
    result = df.copy()
    result["period_key"] = period_key
    return result
