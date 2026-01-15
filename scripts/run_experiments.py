import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cafe_sim.period_key import PeriodKey
from cafe_sim.configuration import ExperimentDefinition, SimulationConfig
from cafe_sim.replication_runner import ReplicationRunner
import yaml


def main():
    project_root = Path(__file__).parent.parent
    calibration_path = project_root / "config" / "period_calibrations.yaml"
    results_dir = project_root / "results"
    
    results_dir.mkdir(exist_ok=True)
    
    print("Café Simulation Experiment Runner")
    print("=" * 80)
    
    runner = ReplicationRunner(calibration_path)
    
    with open(calibration_path, "r") as f:
        calibrations = yaml.safe_load(f)
    
    print("\n1. VALIDATION: Simulating observed periods with baseline configuration")
    print("-" * 80)
    
    validation_results = {}
    
    for period_key in PeriodKey:
        calibration = calibrations[period_key.value]
        
        experiment = ExperimentDefinition(
            name=f"validation_{period_key.value}",
            period_key=period_key,
            barista_count=calibration["suggested_barista_count"],
            run_length_min=calibration["run_length_min"],
            replication_count=30,
            simulation_config=SimulationConfig(
                warmup_min=0.0,
                initial_queue_size=0,
                initial_busy_baristas=0,
                arrival_model_type="poisson",
            ),
            rng_seed=42,
        )
        
        print(f"\n{period_key.to_display_name():20s} | ", end="")
        print(f"m={experiment.barista_count} | ", end="")
        print(f"λ={calibration['poisson_lambda_per_minute']:.3f} | ", end="")
        
        results = runner.run_experiment(experiment)
        validation_results[period_key.value] = results
        
        mean_wait = results.full_period["mean_wait_min"].mean
        utilization = results.full_period["utilization_per_barista"].mean
        
        print(f"Wait={mean_wait:.2f}min | Util={utilization:.2f}")
    
    print("\n\n2. STAFFING SWEEP: Monday Afternoon (m=3, 4, 5)")
    print("-" * 80)
    
    monday_afternoon_calibration = calibrations[PeriodKey.MONDAY_AFTERNOON.value]
    
    staffing_results = {}
    for barista_count in [3, 4, 5]:
        experiment = ExperimentDefinition(
            name=f"staffing_monday_afternoon_m{barista_count}",
            period_key=PeriodKey.MONDAY_AFTERNOON,
            barista_count=barista_count,
            run_length_min=monday_afternoon_calibration["run_length_min"],
            replication_count=30,
            simulation_config=SimulationConfig(
                warmup_min=0.0,
                initial_queue_size=0,
                initial_busy_baristas=0,
                arrival_model_type="poisson",
            ),
            rng_seed=42,
        )
        
        results = runner.run_experiment(experiment)
        staffing_results[barista_count] = results
        
        mean_wait = results.full_period["mean_wait_min"].mean
        wait_ci_width = results.full_period["mean_wait_min"].upper - results.full_period["mean_wait_min"].lower
        utilization = results.full_period["utilization_per_barista"].mean
        p_wait_exceeds_2 = results.full_period["p_wait_exceeds_2min"].mean
        
        print(f"\nm={barista_count}:")
        print(f"  Mean wait: {mean_wait:.2f} ± {wait_ci_width/2:.2f} min")
        print(f"  Utilization: {utilization:.3f}")
        print(f"  P(wait > 2min): {p_wait_exceeds_2:.3f}")
    
    print("\n\n3. SERVICE MIX SENSITIVITY: Varying P(Drink)")
    print("-" * 80)
    
    service_mix_results = {}
    for delta in [-0.15, 0.0, 0.15]:
        experiment = ExperimentDefinition(
            name=f"service_mix_tuesday_morning_delta{delta:+.2f}",
            period_key=PeriodKey.TUESDAY_MORNING,
            barista_count=calibrations[PeriodKey.TUESDAY_MORNING.value]["suggested_barista_count"],
            run_length_min=calibrations[PeriodKey.TUESDAY_MORNING.value]["run_length_min"],
            replication_count=30,
            simulation_config=SimulationConfig(
                warmup_min=0.0,
                initial_queue_size=0,
                initial_busy_baristas=0,
                arrival_model_type="poisson",
            ),
            p_drink_perturbation=delta,
            rng_seed=42,
        )
        
        results = runner.run_experiment(experiment)
        service_mix_results[delta] = results
        
        baseline_p = calibrations[PeriodKey.TUESDAY_MORNING.value]["p_drink"]
        adjusted_p = min(1.0, max(0.0, baseline_p + delta))
        mean_wait = results.full_period["mean_wait_min"].mean
        
        print(f"\nP(Drink) = {adjusted_p:.2f} (baseline {baseline_p:.2f} {delta:+.2f}):")
        print(f"  Mean wait: {mean_wait:.2f} min")
    
    print("\n\n4. WARM-UP SENSITIVITY: Monday Afternoon")
    print("-" * 80)
    
    warmup_results = {}
    for warmup_min in [0, 15, 30]:
        experiment = ExperimentDefinition(
            name=f"warmup_monday_afternoon_{warmup_min}min",
            period_key=PeriodKey.MONDAY_AFTERNOON,
            barista_count=monday_afternoon_calibration["suggested_barista_count"],
            run_length_min=monday_afternoon_calibration["run_length_min"],
            replication_count=30,
            simulation_config=SimulationConfig(
                warmup_min=warmup_min,
                initial_queue_size=0,
                initial_busy_baristas=0,
                arrival_model_type="poisson",
            ),
            rng_seed=42,
        )
        
        results = runner.run_experiment(experiment)
        warmup_results[warmup_min] = results
        
        full_wait = results.full_period["mean_wait_min"].mean
        post_wait = results.post_warmup["mean_wait_min"].mean
        
        print(f"\nWarm-up = {warmup_min} min:")
        print(f"  Mean wait (full): {full_wait:.2f} min")
        print(f"  Mean wait (post): {post_wait:.2f} min")
    
    results_summary_path = results_dir / "experiment_summary.txt"
    with open(results_summary_path, "w") as f:
        f.write("Café Simulation Experiment Results\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. VALIDATION RESULTS\n")
        f.write("-" * 80 + "\n")
        for period_key in PeriodKey:
            results = validation_results[period_key.value]
            f.write(f"\n{period_key.to_display_name()}:\n")
            for kpi_name, ci in results.full_period.items():
                f.write(f"  {kpi_name:30s}: {ci.mean:8.3f} [{ci.lower:7.3f}, {ci.upper:7.3f}]\n")
        
        f.write("\n\n2. STAFFING SWEEP (Monday Afternoon)\n")
        f.write("-" * 80 + "\n")
        for m, results in staffing_results.items():
            f.write(f"\nm = {m} baristas:\n")
            for kpi_name, ci in results.full_period.items():
                f.write(f"  {kpi_name:30s}: {ci.mean:8.3f} [{ci.lower:7.3f}, {ci.upper:7.3f}]\n")
    
    print(f"\n\n✓ Experiments complete! Summary saved to: {results_summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
