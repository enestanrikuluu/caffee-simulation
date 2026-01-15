import numpy as np
from typing import Dict
from cafe_sim.period_key import PeriodKey
from cafe_sim.service_type import ServiceType
from cafe_sim.empirical_distribution import EmpiricalDistribution


class ServiceTypeModel:
    def __init__(
        self,
        p_drink: Dict[PeriodKey, float],
        drink_distributions: Dict[PeriodKey, EmpiricalDistribution],
        food_distributions: Dict[PeriodKey, EmpiricalDistribution],
    ):
        self._p_drink = p_drink
        self._drink_distributions = drink_distributions
        self._food_distributions = food_distributions

    @classmethod
    def from_calibration(cls, period_calibrations: dict) -> "ServiceTypeModel":
        p_drink = {}
        drink_distributions = {}
        food_distributions = {}
        
        for period_key_str, calibration in period_calibrations.items():
            period_key = PeriodKey.from_string(period_key_str)
            
            p_drink[period_key] = calibration["p_drink"]
            
            drink_values = np.array(calibration["empirical_service_drink_min"])
            food_values = np.array(calibration["empirical_service_food_min"])
            
            drink_distributions[period_key] = EmpiricalDistribution(drink_values)
            food_distributions[period_key] = EmpiricalDistribution(food_values)
        
        return cls(p_drink, drink_distributions, food_distributions)

    def sample_service_type(
        self, period_key: PeriodKey, rng: np.random.Generator
    ) -> ServiceType:
        p = self._p_drink[period_key]
        is_drink = rng.uniform(0, 1) < p
        return ServiceType.DRINK if is_drink else ServiceType.DRINK_AND_FOOD

    def sample_service_duration_min(
        self, period_key: PeriodKey, service_type: ServiceType, rng: np.random.Generator
    ) -> float:
        if service_type == ServiceType.DRINK:
            return self._drink_distributions[period_key].sample(rng, size=1)
        else:
            return self._food_distributions[period_key].sample(rng, size=1)

    def create_perturbed_model(self, delta: float) -> "ServiceTypeModel":
        perturbed_p_drink = {}
        for period_key, p in self._p_drink.items():
            new_p = np.clip(p + delta, 0.0, 1.0)
            perturbed_p_drink[period_key] = new_p
        
        return ServiceTypeModel(
            perturbed_p_drink,
            self._drink_distributions,
            self._food_distributions,
        )
