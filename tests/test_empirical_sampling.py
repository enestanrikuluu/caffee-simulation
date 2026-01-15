import pytest
import numpy as np
from cafe_sim.empirical_distribution import EmpiricalDistribution


class TestEmpiricalDistribution:
    def test_initialization_with_valid_data(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dist = EmpiricalDistribution(values)
        assert dist is not None

    def test_initialization_with_empty_array_raises_error(self):
        with pytest.raises(ValueError, match="Cannot create empirical distribution from empty array"):
            EmpiricalDistribution(np.array([]))

    def test_sample_returns_values_in_range(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dist = EmpiricalDistribution(values)
        rng = np.random.default_rng(seed=42)
        
        samples = dist.sample(rng, size=100)
        
        assert len(samples) == 100
        assert np.all(samples >= 1.0)
        assert np.all(samples <= 5.0)

    def test_sample_single_value(self):
        values = np.array([1.0, 2.0, 3.0])
        dist = EmpiricalDistribution(values)
        rng = np.random.default_rng(seed=42)
        
        sample = dist.sample(rng, size=1)
        
        assert isinstance(sample, (float, np.floating))
        assert 1.0 <= sample <= 3.0

    def test_cdf_inversion_at_extremes(self):
        values = np.array([10.0, 20.0, 30.0])
        dist = EmpiricalDistribution(values)
        
        class MockRNG:
            def __init__(self, value):
                self.value = value
            def uniform(self, low, high, size):
                return np.array([self.value])
        
        rng_min = MockRNG(0.0)
        sample_min = dist.sample(rng_min, size=1)
        
        rng_max = MockRNG(1.0)
        sample_max = dist.sample(rng_max, size=1)
        
        assert sample_min >= 10.0
        assert sample_max <= 30.0

    def test_get_percentile(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dist = EmpiricalDistribution(values)
        
        p0 = dist.get_percentile(0)
        p50 = dist.get_percentile(50)
        p100 = dist.get_percentile(100)
        
        assert p0 == 1.0
        assert 2.5 <= p50 <= 3.5
        assert p100 == 5.0

    def test_get_percentile_invalid_raises_error(self):
        values = np.array([1.0, 2.0, 3.0])
        dist = EmpiricalDistribution(values)
        
        with pytest.raises(ValueError, match="Percentile must be between 0 and 100"):
            dist.get_percentile(-10)
        
        with pytest.raises(ValueError, match="Percentile must be between 0 and 100"):
            dist.get_percentile(150)

    def test_get_mean(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dist = EmpiricalDistribution(values)
        
        mean = dist.get_mean()
        
        assert mean == 3.0

    def test_get_std(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dist = EmpiricalDistribution(values)
        
        std = dist.get_std()
        
        assert std > 0
        assert np.isclose(std, np.std(values))

    def test_reproducibility_with_seed(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dist = EmpiricalDistribution(values)
        
        rng1 = np.random.default_rng(seed=123)
        samples1 = dist.sample(rng1, size=10)
        
        rng2 = np.random.default_rng(seed=123)
        samples2 = dist.sample(rng2, size=10)
        
        assert np.allclose(samples1, samples2)
