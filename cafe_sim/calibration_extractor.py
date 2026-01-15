import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from sklearn.cluster import KMeans
from cafe_sim.period_key import PeriodKey


@dataclass
class ThresholdDiagnostic:
    threshold: float
    p_drink: float
    mean_drink: float
    mean_food: float
    mean_difference: float
    overlap_coefficient: float


class ThresholdDiagnosticAnalyzer:
    def analyze_thresholds(
        self, service_times: np.ndarray, thresholds: List[float]
    ) -> List[ThresholdDiagnostic]:
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
            
            overlap = self._compute_overlap_coefficient(drink_times, food_times)
            
            diagnostics.append(
                ThresholdDiagnostic(
                    threshold=threshold,
                    p_drink=p_drink,
                    mean_drink=mean_drink,
                    mean_food=mean_food,
                    mean_difference=mean_diff,
                    overlap_coefficient=overlap,
                )
            )
        
        return diagnostics

    def analyze_kmeans_clustering(
        self, service_times: np.ndarray, jitter_amount: float = 0.5
    ) -> ThresholdDiagnostic:
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
        overlap = self._compute_overlap_coefficient(drink_times, food_times)
        
        return ThresholdDiagnostic(
            threshold=threshold,
            p_drink=p_drink,
            mean_drink=cluster_means[drink_label],
            mean_food=cluster_means[food_label],
            mean_difference=cluster_means[food_label] - cluster_means[drink_label],
            overlap_coefficient=overlap,
        )

    def _compute_overlap_coefficient(
        self, group1: np.ndarray, group2: np.ndarray
    ) -> float:
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


class CalibrationExtractor:
    def __init__(self, threshold_analyzer: ThresholdDiagnosticAnalyzer):
        self._threshold_analyzer = threshold_analyzer

    def extract_period_statistics(
        self, cleaned_data: Dict[PeriodKey, pd.DataFrame], baseline_threshold: float = 1.0
    ) -> Dict[PeriodKey, Dict]:
        statistics = {}
        
        for period_key, df in cleaned_data.items():
            period_stats = self._extract_single_period(df, period_key, baseline_threshold)
            statistics[period_key] = period_stats
        
        return statistics

    def generate_threshold_diagnostics(
        self, cleaned_data: Dict[PeriodKey, pd.DataFrame]
    ) -> Dict[PeriodKey, Dict]:
        diagnostics = {}
        
        test_thresholds = [1.0, 1.5, 2.0]
        
        for period_key, df in cleaned_data.items():
            if "service_time_min" not in df.columns:
                continue
            
            service_times = df["service_time_min"].dropna().values
            
            if len(service_times) < 10:
                continue
            
            threshold_diagnostics = self._threshold_analyzer.analyze_thresholds(
                service_times, test_thresholds
            )
            
            try:
                kmeans_diagnostic = self._threshold_analyzer.analyze_kmeans_clustering(
                    service_times
                )
            except Exception:
                kmeans_diagnostic = None
            
            diagnostics[period_key] = {
                "threshold_splits": threshold_diagnostics,
                "kmeans_split": kmeans_diagnostic,
            }
        
        return diagnostics

    def _extract_single_period(
        self, df: pd.DataFrame, period_key: PeriodKey, threshold: float
    ) -> Dict:
        service_times = df["service_time_min"].dropna().values
        
        drink_times = service_times[service_times <= threshold]
        food_times = service_times[service_times > threshold]
        
        p_drink = len(drink_times) / len(service_times) if len(service_times) > 0 else 0.5
        
        stats = {
            "period_key": period_key.value,
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
