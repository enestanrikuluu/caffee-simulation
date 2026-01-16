import numpy as np
from scipy import stats
from cafe_sim.configuration import create_confidence_interval, create_aggregated_kpis


def aggregate_kpis(full_period_kpis, post_warmup_kpis, alpha=0.05):
    full_ci = compute_kpi_confidence_intervals(full_period_kpis, alpha)
    post_ci = compute_kpi_confidence_intervals(post_warmup_kpis, alpha)

    return create_aggregated_kpis(full_period=full_ci, post_warmup=post_ci)


def compute_kpi_confidence_intervals(kpis, alpha):
    if len(kpis) == 0:
        return {}

    kpi_names = [
        "mean_wait_min",
        "median_wait_min",
        "p90_wait_min",
        "p95_wait_min",
        "time_avg_queue_length",
        "utilization_per_barista",
        "p_wait_exceeds_2min",
        "throughput",
    ]

    results = {}
    for kpi_name in kpi_names:
        values = np.array([kpi[kpi_name] for kpi in kpis])
        ci = compute_confidence_interval(values, alpha)
        results[kpi_name] = ci

    return results


def compute_confidence_interval(values, alpha):
    n = len(values)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0

    if n > 1:
        t_critical = stats.t.ppf(1 - alpha / 2, df=n - 1)
        margin = t_critical * std / np.sqrt(n)
        lower = mean - margin
        upper = mean + margin
    else:
        lower = mean
        upper = mean

    return create_confidence_interval(
        mean=mean,
        std=std,
        lower=lower,
        upper=upper,
        alpha=alpha,
        sample_size=n,
    )
