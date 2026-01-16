import numpy as np
import yaml
from pathlib import Path
from cafe_sim.simulation_engine import create_simulation_engine, run_simulation
from cafe_sim.configuration import create_aggregated_kpis
from cafe_sim.kpi_aggregation import aggregate_kpis
from cafe_sim.arrival_process_model import create_arrival_model
from cafe_sim.service_type_model import create_service_model_from_calibration, create_perturbed_service_model


def load_period_calibrations(period_calibrations_path):
    with open(period_calibrations_path, "r") as f:
        return yaml.safe_load(f)


def run_experiment(period_calibrations, experiment):
    full_period_kpis = []
    post_warmup_kpis = []

    arrival_model = create_arrival_model(
        experiment['simulation_config']['arrival_model_type'],
        period_calibrations,
    )

    service_model = create_service_model_from_calibration(period_calibrations)

    if experiment['p_drink_perturbation'] != 0.0:
        service_model = create_perturbed_service_model(
            service_model,
            experiment['p_drink_perturbation']
        )

    base_seed = experiment['rng_seed'] if experiment['rng_seed'] is not None else 42

    for replication_idx in range(experiment['replication_count']):
        replication_seed = base_seed + replication_idx

        full_kpi, post_kpi = run_single_replication(
            experiment=experiment,
            arrival_model=arrival_model,
            service_model=service_model,
            seed=replication_seed,
        )

        full_period_kpis.append(full_kpi)
        post_warmup_kpis.append(post_kpi)

    return aggregate_kpis(full_period_kpis, post_warmup_kpis)


def run_single_replication(experiment, arrival_model, service_model, seed):
    engine = create_simulation_engine(
        period_key=experiment['period_key'],
        barista_count=experiment['barista_count'],
        run_length_min=experiment['run_length_min'],
        arrival_model=arrival_model,
        service_model=service_model,
        simulation_config=experiment['simulation_config'],
    )

    return run_simulation(engine, rng_seed=seed)
