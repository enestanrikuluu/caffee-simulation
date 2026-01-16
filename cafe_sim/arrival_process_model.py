import numpy as np
from cafe_sim.period_key import period_key_from_string
from cafe_sim.empirical_distribution import create_empirical_distribution, sample_from_empirical


def create_poisson_arrival_model(lambda_per_minute):
    return {
        'type': 'poisson',
        'lambda_per_minute': lambda_per_minute,
    }


def create_empirical_arrival_model(empirical_distributions):
    distributions = {}
    for period_key, interarrival_values in empirical_distributions.items():
        if len(interarrival_values) > 0:
            distributions[period_key] = create_empirical_distribution(
                np.array(interarrival_values)
            )
    return {
        'type': 'empirical',
        'distributions': distributions,
    }


def sample_poisson_arrival_delay(model, period_key, current_time_min, rng):
    lambda_rate = model['lambda_per_minute'][period_key]

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


def sample_empirical_arrival_delay(model, period_key, current_time_min, rng):
    if period_key not in model['distributions']:
        return 1.0

    empirical_sample = sample_from_empirical(model['distributions'][period_key], rng, size=1)
    jitter = rng.uniform(0, 1)

    return float(empirical_sample) + jitter


def sample_next_arrival_delay(model, period_key, current_time_min, rng):
    if model['type'] == 'poisson':
        return sample_poisson_arrival_delay(model, period_key, current_time_min, rng)
    elif model['type'] == 'empirical':
        return sample_empirical_arrival_delay(model, period_key, current_time_min, rng)
    else:
        raise ValueError(f"Unknown arrival model type: {model['type']}")


def create_arrival_model(model_type, period_calibrations):
    if model_type == "poisson":
        lambda_dict = {}
        for period_key_str, calibration in period_calibrations.items():
            period_key = period_key_from_string(period_key_str)
            lambda_dict[period_key] = calibration["poisson_lambda_per_minute"]
        return create_poisson_arrival_model(lambda_dict)

    elif model_type == "empirical":
        empirical_dict = {}
        for period_key_str, calibration in period_calibrations.items():
            period_key = period_key_from_string(period_key_str)
            empirical_dict[period_key] = calibration["empirical_interarrivals_min"]
        return create_empirical_arrival_model(empirical_dict)

    else:
        raise ValueError(f"Unknown arrival model type: {model_type}")
