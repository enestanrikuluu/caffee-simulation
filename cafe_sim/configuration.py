from dataclasses import dataclass, field
from typing import Dict, List, Optional
from cafe_sim.period_key import PeriodKey


@dataclass
class PeriodCalibration:
    period_key: PeriodKey
    run_length_min: float
    barista_count: int
    arrival_rate_per_hour: float
    poisson_lambda_per_minute: float
    p_drink: float
    empirical_interarrivals_min: List[float]
    empirical_service_drink_min: List[float]
    empirical_service_food_min: List[float]


@dataclass
class SimulationConfig:
    warmup_min: float = 0.0
    initial_queue_size: int = 0
    initial_busy_baristas: int = 0
    arrival_model_type: str = "poisson"


@dataclass
class ExperimentDefinition:
    name: str
    period_key: PeriodKey
    barista_count: int
    simulation_config: SimulationConfig
    run_length_min: float
    replication_count: int
    p_drink_perturbation: float = 0.0
    rng_seed: Optional[int] = None


@dataclass
class ConfidenceInterval:
    mean: float
    std: float
    lower: float
    upper: float
    alpha: float
    sample_size: int


@dataclass
class KPISet:
    mean_wait_min: float
    median_wait_min: float
    p90_wait_min: float
    p95_wait_min: float
    mean_queue_length: float
    time_avg_queue_length: float
    utilization_per_barista: float
    p_wait_exceeds_2min: float
    throughput: int
    mean_system_time_min: float = 0.0


@dataclass
class AggregatedKPIs:
    full_period: Dict[str, ConfidenceInterval] = field(default_factory=dict)
    post_warmup: Dict[str, ConfidenceInterval] = field(default_factory=dict)
