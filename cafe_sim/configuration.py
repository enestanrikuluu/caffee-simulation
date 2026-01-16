def create_period_calibration(
    period_key,
    run_length_min,
    barista_count,
    arrival_rate_per_hour,
    poisson_lambda_per_minute,
    p_drink,
    empirical_interarrivals_min,
    empirical_service_drink_min,
    empirical_service_food_min
):
    return {
        'period_key': period_key,
        'run_length_min': run_length_min,
        'barista_count': barista_count,
        'arrival_rate_per_hour': arrival_rate_per_hour,
        'poisson_lambda_per_minute': poisson_lambda_per_minute,
        'p_drink': p_drink,
        'empirical_interarrivals_min': empirical_interarrivals_min,
        'empirical_service_drink_min': empirical_service_drink_min,
        'empirical_service_food_min': empirical_service_food_min,
    }


def create_simulation_config(
    warmup_min=0.0,
    initial_queue_size=0,
    initial_busy_baristas=0,
    arrival_model_type="poisson"
):
    return {
        'warmup_min': warmup_min,
        'initial_queue_size': initial_queue_size,
        'initial_busy_baristas': initial_busy_baristas,
        'arrival_model_type': arrival_model_type,
    }


def create_experiment_definition(
    name,
    period_key,
    barista_count,
    simulation_config,
    run_length_min,
    replication_count,
    p_drink_perturbation=0.0,
    rng_seed=None
):
    return {
        'name': name,
        'period_key': period_key,
        'barista_count': barista_count,
        'simulation_config': simulation_config,
        'run_length_min': run_length_min,
        'replication_count': replication_count,
        'p_drink_perturbation': p_drink_perturbation,
        'rng_seed': rng_seed,
    }


def create_confidence_interval(mean, std, lower, upper, alpha, sample_size):
    return {
        'mean': mean,
        'std': std,
        'lower': lower,
        'upper': upper,
        'alpha': alpha,
        'sample_size': sample_size,
    }


def create_kpi_set(
    mean_wait_min,
    median_wait_min,
    p90_wait_min,
    p95_wait_min,
    mean_queue_length,
    time_avg_queue_length,
    utilization_per_barista,
    p_wait_exceeds_2min,
    throughput,
    mean_system_time_min=0.0
):
    return {
        'mean_wait_min': mean_wait_min,
        'median_wait_min': median_wait_min,
        'p90_wait_min': p90_wait_min,
        'p95_wait_min': p95_wait_min,
        'mean_queue_length': mean_queue_length,
        'time_avg_queue_length': time_avg_queue_length,
        'utilization_per_barista': utilization_per_barista,
        'p_wait_exceeds_2min': p_wait_exceeds_2min,
        'throughput': throughput,
        'mean_system_time_min': mean_system_time_min,
    }


def create_aggregated_kpis(full_period=None, post_warmup=None):
    return {
        'full_period': full_period if full_period is not None else {},
        'post_warmup': post_warmup if post_warmup is not None else {},
    }
