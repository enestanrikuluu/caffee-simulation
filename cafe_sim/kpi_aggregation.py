import numpy as np
from scipy import stats
from typing import List
from cafe_sim.configuration import KPISet, ConfidenceInterval, AggregatedKPIs


class KPIAggregator:
    def aggregate_kpis(
        self, 
        full_period_kpis: List[KPISet],
        post_warmup_kpis: List[KPISet],
        alpha: float = 0.05
    ) -> AggregatedKPIs:
        full_ci = self._compute_kpi_confidence_intervals(full_period_kpis, alpha)
        post_ci = self._compute_kpi_confidence_intervals(post_warmup_kpis, alpha)
        
        return AggregatedKPIs(full_period=full_ci, post_warmup=post_ci)

    def _compute_kpi_confidence_intervals(
        self, kpis: List[KPISet], alpha: float
    ) -> dict:
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
            values = np.array([getattr(kpi, kpi_name) for kpi in kpis])
            ci = self._compute_confidence_interval(values, alpha)
            results[kpi_name] = ci
        
        return results

    def _compute_confidence_interval(
        self, values: np.ndarray, alpha: float
    ) -> ConfidenceInterval:
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
        
        return ConfidenceInterval(
            mean=mean,
            std=std,
            lower=lower,
            upper=upper,
            alpha=alpha,
            sample_size=n,
        )
