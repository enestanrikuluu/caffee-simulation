import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


def create_threshold_diagnostic(threshold, p_drink, mean_drink, mean_food, mean_difference, overlap_coefficient):
    return {
        'threshold': threshold,
        'p_drink': p_drink,
        'mean_drink': mean_drink,
        'mean_food': mean_food,
        'mean_difference': mean_difference,
        'overlap_coefficient': overlap_coefficient,
    }


def analyze_thresholds(service_times, thresholds):
    diagnostics = []

    for threshold in thresholds:
        drink_times = service_times[service_times <= threshold]
        food_times = service_times[service_times > threshold]

        if len(drink_times) == 0 or len(food_times) == 0:
            continue

        p_drink = len(drink_times) / len(service_times)
        mean_drink = float(np.mean(drink_times))
        mean_food = float(np.mean(food_times))
        mean_diff = mean_food - mean_drink

        overlap = compute_overlap_coefficient(drink_times, food_times)

        diagnostics.append(
            create_threshold_diagnostic(
                threshold=threshold,
                p_drink=p_drink,
                mean_drink=mean_drink,
                mean_food=mean_food,
                mean_difference=mean_diff,
                overlap_coefficient=overlap,
            )
        )

    return diagnostics


def analyze_kmeans_clustering(service_times, jitter_amount=0.5):
    jittered_times = service_times + np.random.uniform(
        -jitter_amount, jitter_amount, size=len(service_times)
    )

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(jittered_times.reshape(-1, 1))

    cluster_means = [
        float(np.mean(service_times[labels == i])) for i in range(2)
    ]
    drink_label = 0 if cluster_means[0] < cluster_means[1] else 1
    food_label = 1 - drink_label

    drink_times = service_times[labels == drink_label]
    food_times = service_times[labels == food_label]

    threshold = (cluster_means[drink_label] + cluster_means[food_label]) / 2
    p_drink = len(drink_times) / len(service_times)
    overlap = compute_overlap_coefficient(drink_times, food_times)

    return create_threshold_diagnostic(
        threshold=threshold,
        p_drink=p_drink,
        mean_drink=cluster_means[drink_label],
        mean_food=cluster_means[food_label],
        mean_difference=cluster_means[food_label] - cluster_means[drink_label],
        overlap_coefficient=overlap,
    )


def compute_overlap_coefficient(group1, group2):
    min_max1 = (np.min(group1), np.max(group1))
    min_max2 = (np.min(group2), np.max(group2))

    overlap_start = max(min_max1[0], min_max2[0])
    overlap_end = min(min_max1[1], min_max2[1])

    if overlap_start >= overlap_end:
        return 0.0

    overlap_length = overlap_end - overlap_start
    range1_length = min_max1[1] - min_max1[0]
    range2_length = min_max2[1] - min_max2[0]

    min_range = min(range1_length, range2_length)

    if min_range == 0:
        return 0.0

    return overlap_length / min_range


def extract_period_statistics(cleaned_data, baseline_threshold=1.0):
    statistics = {}

    for period_key, df in cleaned_data.items():
        period_stats = extract_single_period(df, period_key, baseline_threshold)
        statistics[period_key] = period_stats

    return statistics


def generate_threshold_diagnostics(cleaned_data):
    diagnostics = {}

    test_thresholds = [1.0, 1.5, 2.0]

    for period_key, df in cleaned_data.items():
        if "service_time_min" not in df.columns:
            continue

        service_times = df["service_time_min"].dropna().values

        if len(service_times) < 10:
            continue

        threshold_diagnostics = analyze_thresholds(
            service_times, test_thresholds
        )

        try:
            kmeans_diagnostic = analyze_kmeans_clustering(service_times)
        except Exception:
            kmeans_diagnostic = None

        diagnostics[period_key] = {
            "threshold_splits": threshold_diagnostics,
            "kmeans_split": kmeans_diagnostic,
        }

    return diagnostics


def extract_single_period(df, period_key, threshold):
    # Use service_type column from data if available, otherwise fall back to threshold
    if "service_type" in df.columns:
        # Use the labels from the data
        drink_mask = df["service_type"] == "drink"
        food_mask = df["service_type"] == "food"

        drink_times = df.loc[drink_mask, "service_time_min"].dropna().values
        food_times = df.loc[food_mask, "service_time_min"].dropna().values

        total_labeled = len(drink_times) + len(food_times)
        p_drink = len(drink_times) / total_labeled if total_labeled > 0 else 0.5
    else:
        # Fall back to threshold-based approach
        service_times = df["service_time_min"].dropna().values
        drink_times = service_times[service_times <= threshold]
        food_times = service_times[service_times > threshold]
        p_drink = len(drink_times) / len(service_times) if len(service_times) > 0 else 0.5

    stats = {
        "period_key": period_key,
        "observation_count": len(df),
        "p_drink": p_drink,
        "empirical_service_drink_min": drink_times.tolist() if len(drink_times) > 0 else [1.0],
        "empirical_service_food_min": food_times.tolist() if len(food_times) > 0 else [2.0],
        "mean_service_drink": float(np.mean(drink_times)) if len(drink_times) > 0 else 1.0,
        "mean_service_food": float(np.mean(food_times)) if len(food_times) > 0 else 2.0,
    }

    if "arrival_time" in df.columns:
        arrival_times = df["arrival_time"].dropna().values
        if len(arrival_times) > 0:
            span_min = float(np.max(arrival_times) - np.min(arrival_times))
            stats["run_length_min"] = span_min
            stats["arrival_rate_per_hour"] = (len(arrival_times) / span_min) * 60 if span_min > 0 else 0
            stats["poisson_lambda_per_minute"] = len(arrival_times) / span_min if span_min > 0 else 0

    if "interarrival_min" in df.columns:
        interarrivals = df["interarrival_min"].dropna().values
        stats["empirical_interarrivals_min"] = interarrivals.tolist()

        if "run_length_min" not in stats and len(interarrivals) > 0:
            estimated_span = float(np.sum(interarrivals))
            stats["run_length_min"] = estimated_span
            stats["arrival_rate_per_hour"] = (len(interarrivals) / estimated_span) * 60 if estimated_span > 0 else 0
            stats["poisson_lambda_per_minute"] = len(interarrivals) / estimated_span if estimated_span > 0 else 0

    if "wait_min" in df.columns:
        wait_times = df["wait_min"].dropna().values
        if len(wait_times) > 0:
            stats["mean_wait_min"] = float(np.mean(wait_times))
            stats["p_wait_positive"] = float(np.mean(wait_times > 0))

    return stats
