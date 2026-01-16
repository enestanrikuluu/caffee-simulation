import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cafe_sim.data_loading import load_all_periods
from cafe_sim.data_cleaning import clean_data
from cafe_sim.calibration_extractor import generate_threshold_diagnostics
from cafe_sim.period_key import ALL_PERIOD_KEYS, MONDAY_AFTERNOON, TUESDAY_MORNING, period_key_to_display_name
from cafe_sim.configuration import create_experiment_definition, create_simulation_config
from cafe_sim.replication_runner import load_period_calibrations, run_experiment
from cafe_sim import visualization as viz
import yaml


def main():
    project_root = Path(__file__).parent.parent
    data_directory = project_root / "simulation_data"
    calibration_path = project_root / "config" / "period_calibrations.yaml"
    results_dir = project_root / "results"
    plots_dir = results_dir / "plots"

    plots_dir.mkdir(parents=True, exist_ok=True)

    print("Cafe Simulation Visualization Pipeline")
    print("=" * 80)

    print("\n1. Loading and cleaning observation data...")
    raw_data = load_all_periods(data_directory)

    cleaned_data = {
        period_key: clean_data(raw_df, period_key)
        for period_key, raw_df in raw_data.items()
    }

    print(" Data loaded")

    print("\n2. Extracting calibration diagnostics...")
    diagnostics = generate_threshold_diagnostics(cleaned_data)

    with open(calibration_path, "r") as f:
        calibrations = yaml.safe_load(f)

    period_calibrations = load_period_calibrations(calibration_path)

    print(" Diagnostics extracted")

    print("\n3. Generating calibration plots...")

    service_hist_path = viz.plot_calibration_service_histograms(cleaned_data, plots_dir, threshold=1.0)
    print(f"    Service histograms: {service_hist_path.name}")

    interarrival_path = viz.plot_calibration_interarrival_ecdfs(cleaned_data, plots_dir)
    print(f"    Interarrival ECDFs: {interarrival_path.name}")

    threshold_diag_path = viz.plot_threshold_diagnostics(diagnostics, 1.0, plots_dir)
    print(f"    Threshold diagnostics: {threshold_diag_path.name}")

    print("\n4. Running validation experiments...")

    validation_results = {}
    observed_stats = {}

    for period_key in ALL_PERIOD_KEYS:
        calibration = calibrations[period_key]

        experiment = create_experiment_definition(
            name=f"validation_{period_key}",
            period_key=period_key,
            barista_count=calibration["suggested_barista_count"],
            run_length_min=calibration["run_length_min"],
            replication_count=30,
            simulation_config=create_simulation_config(
                warmup_min=0.0,
                arrival_model_type="poisson",
            ),
            rng_seed=42,
        )

        results = run_experiment(period_calibrations, experiment)
        validation_results[period_key] = results

        observed_stats[period_key] = {
            'mean_wait_min': 0.0,
            'utilization_per_barista': 0.0,
            'throughput': len(cleaned_data[period_key]),
        }

    print("    Validation complete")

    print("\n5. Generating validation plots...")

    validation_path = viz.plot_validation_kpi_comparison(validation_results, observed_stats, plots_dir)
    print(f"    KPI comparison: {validation_path.name}")

    wait_ecdf_path = viz.plot_wait_time_ecdf_comparison(validation_results, cleaned_data, plots_dir)
    print(f"    Wait ECDF comparison: {wait_ecdf_path.name}")

    print("\n6. Running staffing sweep (Monday Afternoon)...")
    staffing_results = {}
    monday_afternoon_calibration = calibrations[MONDAY_AFTERNOON]

    for barista_count in [3, 4, 5]:
        experiment = create_experiment_definition(
            name=f"staffing_monday_afternoon_m{barista_count}",
            period_key=MONDAY_AFTERNOON,
            barista_count=barista_count,
            run_length_min=monday_afternoon_calibration["run_length_min"],
            replication_count=30,
            simulation_config=create_simulation_config(
                warmup_min=0.0,
                arrival_model_type="poisson",
            ),
            rng_seed=42,
        )

        results = run_experiment(period_calibrations, experiment)
        staffing_results[barista_count] = results

    staffing_path = viz.plot_staffing_sweep(staffing_results, "Monday Afternoon", plots_dir)
    print(f"    Staffing sweep: {staffing_path.name}")

    print("\n7. Running service mix sensitivity (Tuesday Morning)...")
    service_mix_results = {}
    tuesday_morning_calibration = calibrations[TUESDAY_MORNING]
    baseline_p_drink = tuesday_morning_calibration["p_drink"]

    for delta in [-0.15, 0.0, 0.15]:
        experiment = create_experiment_definition(
            name=f"service_mix_tuesday_morning_delta{delta:+.2f}",
            period_key=TUESDAY_MORNING,
            barista_count=tuesday_morning_calibration["suggested_barista_count"],
            run_length_min=tuesday_morning_calibration["run_length_min"],
            replication_count=30,
            simulation_config=create_simulation_config(
                warmup_min=0.0,
                arrival_model_type="poisson",
            ),
            p_drink_perturbation=delta,
            rng_seed=42,
        )

        results = run_experiment(period_calibrations, experiment)
        service_mix_results[delta] = results

    service_mix_path = viz.plot_service_mix_sensitivity(
        service_mix_results, baseline_p_drink, "Tuesday Morning", plots_dir
    )
    print(f"    Service mix sensitivity: {service_mix_path.name}")

    print("\n8. Running warm-up analysis (Monday Afternoon)...")
    warmup_results = {}

    for warmup_min in [0, 15, 30]:
        experiment = create_experiment_definition(
            name=f"warmup_monday_afternoon_{warmup_min}min",
            period_key=MONDAY_AFTERNOON,
            barista_count=monday_afternoon_calibration["suggested_barista_count"],
            run_length_min=monday_afternoon_calibration["run_length_min"],
            replication_count=30,
            simulation_config=create_simulation_config(
                warmup_min=warmup_min,
                arrival_model_type="poisson",
            ),
            rng_seed=42,
        )

        results = run_experiment(period_calibrations, experiment)
        warmup_results[warmup_min] = results

    warmup_path = viz.plot_warmup_comparison(warmup_results, "Monday Afternoon", plots_dir)
    print(f"    Warm-up analysis: {warmup_path.name}")

    print("\n" + "=" * 80)
    print(f" All visualizations generated successfully!")
    print(f"  Output directory: {plots_dir}")
    print(f"  Total plots: 8")
    print("=" * 80)


if __name__ == "__main__":
    main()
