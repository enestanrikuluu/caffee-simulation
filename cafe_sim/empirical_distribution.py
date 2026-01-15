import numpy as np
from typing import Tuple


class EmpiricalDistribution:
    def __init__(self, values: np.ndarray):
        if len(values) == 0:
            raise ValueError("Cannot create empirical distribution from empty array")
        
        self._sorted_values, self._cdf_values = self._build_cdf(values)

    def sample(self, rng: np.random.Generator, size: int = 1) -> np.ndarray:
        uniform_samples = rng.uniform(0, 1, size=size)
        sampled_values = np.interp(
            uniform_samples,
            self._cdf_values,
            self._sorted_values
        )
        return sampled_values if size > 1 else sampled_values[0]

    def _build_cdf(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        sorted_values = np.sort(values)
        n = len(sorted_values)
        cdf_values = np.arange(1, n + 1) / n
        return sorted_values, cdf_values

    def get_percentile(self, percentile: float) -> float:
        if not 0 <= percentile <= 100:
            raise ValueError("Percentile must be between 0 and 100")
        
        cdf_target = percentile / 100.0
        return np.interp(cdf_target, self._cdf_values, self._sorted_values)

    def get_mean(self) -> float:
        return float(np.mean(self._sorted_values))

    def get_std(self) -> float:
        return float(np.std(self._sorted_values))
