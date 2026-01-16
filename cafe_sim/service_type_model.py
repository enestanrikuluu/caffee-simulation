import numpy as np
from cafe_sim.period_key import period_key_from_string
from cafe_sim.service_type import DRINK, DRINK_AND_FOOD
from cafe_sim.empirical_distribution import create_empirical_distribution, sample_from_empirical


def create_service_model(p_drink, drink_distributions, food_distributions):
    return {
        'p_drink': p_drink,
        'drink_distributions': drink_distributions,
        'food_distributions': food_distributions,
    }


def create_service_model_from_calibration(period_calibrations):
    p_drink = {}
    drink_distributions = {}
    food_distributions = {}

    for period_key_str, calibration in period_calibrations.items():
        period_key = period_key_from_string(period_key_str)

        p_drink[period_key] = calibration["p_drink"]

        drink_values = np.array(calibration["empirical_service_drink_min"])
        food_values = np.array(calibration["empirical_service_food_min"])

        drink_distributions[period_key] = create_empirical_distribution(drink_values)
        food_distributions[period_key] = create_empirical_distribution(food_values)

    return create_service_model(p_drink, drink_distributions, food_distributions)


def sample_service_type(model, period_key, rng):
    p = model['p_drink'][period_key]
    is_drink = rng.uniform(0, 1) < p
    return DRINK if is_drink else DRINK_AND_FOOD


def sample_service_duration(model, period_key, service_type, rng):
    if service_type == DRINK:
        return sample_from_empirical(model['drink_distributions'][period_key], rng, size=1)
    else:
        return sample_from_empirical(model['food_distributions'][period_key], rng, size=1)


def create_perturbed_service_model(model, delta):
    perturbed_p_drink = {}
    for period_key, p in model['p_drink'].items():
        new_p = np.clip(p + delta, 0.0, 1.0)
        perturbed_p_drink[period_key] = new_p

    return create_service_model(
        perturbed_p_drink,
        model['drink_distributions'],
        model['food_distributions'],
    )
