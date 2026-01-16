import numpy as np


def create_empirical_distribution(values):
    if len(values) == 0:
        raise ValueError("Cannot create empirical distribution from empty array")

    sorted_values = np.sort(values)
    n = len(sorted_values)
    cdf_values = np.arange(1, n + 1) / n

    return {
        'sorted_values': sorted_values,
        'cdf_values': cdf_values,
    }


def sample_from_empirical(dist, rng, size=1):
    uniform_samples = rng.uniform(0, 1, size=size)
    sampled_values = np.interp(
        uniform_samples,
        dist['cdf_values'],
        dist['sorted_values']
    )
    return sampled_values if size > 1 else sampled_values[0]


def get_empirical_percentile(dist, percentile):
    if not 0 <= percentile <= 100:
        raise ValueError("Percentile must be between 0 and 100")

    cdf_target = percentile / 100.0
    return np.interp(cdf_target, dist['cdf_values'], dist['sorted_values'])


def get_empirical_mean(dist):
    return float(np.mean(dist['sorted_values']))


def get_empirical_std(dist):
    return float(np.std(dist['sorted_values']))
