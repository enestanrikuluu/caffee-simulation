from abc import ABC, abstractmethod
import numpy as np
from cafe_sim.period_key import PeriodKey


class ArrivalProcessModel(ABC):
    @abstractmethod
    def sample_next_arrival_delay_min(
        self, period_key: PeriodKey, current_time_min: float, rng: np.random.Generator
    ) -> float:
        pass


class PoissonPerMinuteArrivalModel(ArrivalProcessModel):
    def __init__(self, lambda_per_minute: dict[PeriodKey, float]):
        self._lambda_per_minute = lambda_per_minute

    def sample_next_arrival_delay_min(
        self, period_key: PeriodKey, current_time_min: float, rng: np.random.Generator
    ) -> float:
        lambda_rate = self._lambda_per_minute[period_key]
        
        current_minute_bucket = int(current_time_min)
        time_in_bucket = current_time_min - current_minute_bucket
        
        if time_in_bucket == 0:
            arrivals_this_minute = rng.poisson(lambda_rate)
            
            if arrivals_this_minute > 0:
                uniform_offset = rng.uniform(0, 1)
                return uniform_offset
            else:
                return 1.0
        else:
            remaining_in_bucket = 1.0 - time_in_bucket
            return remaining_in_bucket + rng.uniform(0, 1)


class EmpiricalInterarrivalModel(ArrivalProcessModel):
    def __init__(self, empirical_distributions: dict):
        from cafe_sim.empirical_distribution import EmpiricalDistribution
        
        self._distributions = {}
        for period_key, interarrival_values in empirical_distributions.items():
            if len(interarrival_values) > 0:
                self._distributions[period_key] = EmpiricalDistribution(
                    np.array(interarrival_values)
                )

    def sample_next_arrival_delay_min(
        self, period_key: PeriodKey, current_time_min: float, rng: np.random.Generator
    ) -> float:
        if period_key not in self._distributions:
            return 1.0
        
        empirical_sample = self._distributions[period_key].sample(rng, size=1)
        jitter = rng.uniform(0, 1)
        
        return float(empirical_sample) + jitter


class ArrivalModelFactory:
    @staticmethod
    def create(
        model_type: str,
        period_calibrations: dict,
    ):
        if model_type == "poisson":
            lambda_dict = {}
            for period_key_str, calibration in period_calibrations.items():
                period_key = PeriodKey.from_string(period_key_str)
                lambda_dict[period_key] = calibration["poisson_lambda_per_minute"]
            return PoissonPerMinuteArrivalModel(lambda_dict)
        
        elif model_type == "empirical":
            empirical_dict = {}
            for period_key_str, calibration in period_calibrations.items():
                period_key = PeriodKey.from_string(period_key_str)
                empirical_dict[period_key] = calibration["empirical_interarrivals_min"]
            return EmpiricalInterarrivalModel(empirical_dict)
        
        else:
            raise ValueError(f"Unknown arrival model type: {model_type}")
