from typing import List
import numpy as np
from cafe_sim.simulation_engine import CafeSimulationEngine
from cafe_sim.configuration import ExperimentDefinition, KPISet, AggregatedKPIs
from cafe_sim.kpi_aggregation import KPIAggregator
from cafe_sim.arrival_process_model import ArrivalModelFactory
from cafe_sim.service_type_model import ServiceTypeModel
import yaml
from pathlib import Path


class ReplicationRunner:
    def __init__(self, period_calibrations_path: Path):
        with open(period_calibrations_path, "r") as f:
            self._period_calibrations = yaml.safe_load(f)
    
    def run_experiment(self, experiment: ExperimentDefinition) -> AggregatedKPIs:
        full_period_kpis: List[KPISet] = []
        post_warmup_kpis: List[KPISet] = []
        
        arrival_model = ArrivalModelFactory.create(
            experiment.simulation_config.arrival_model_type,
            self._period_calibrations,
        )
        
        service_model = ServiceTypeModel.from_calibration(self._period_calibrations)
        
        if experiment.p_drink_perturbation != 0.0:
            service_model = service_model.create_perturbed_model(
                experiment.p_drink_perturbation
            )
        
        base_seed = experiment.rng_seed if experiment.rng_seed is not None else 42
        
        for replication_idx in range(experiment.replication_count):
            replication_seed = base_seed + replication_idx
            
            full_kpi, post_kpi = self._run_single_replication(
                experiment=experiment,
                arrival_model=arrival_model,
                service_model=service_model,
                seed=replication_seed,
            )
            
            full_period_kpis.append(full_kpi)
            post_warmup_kpis.append(post_kpi)
        
        aggregator = KPIAggregator()
        return aggregator.aggregate_kpis(full_period_kpis, post_warmup_kpis)

    def _run_single_replication(
        self,
        experiment: ExperimentDefinition,
        arrival_model,
        service_model,
        seed: int,
    ) -> tuple[KPISet, KPISet]:
        engine = CafeSimulationEngine(
            period_key=experiment.period_key,
            barista_count=experiment.barista_count,
            run_length_min=experiment.run_length_min,
            arrival_model=arrival_model,
            service_model=service_model,
            simulation_config=experiment.simulation_config,
        )
        
        return engine.run(rng_seed=seed)
