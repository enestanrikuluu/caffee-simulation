import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cafe_sim.data_loading import load_all_periods
from cafe_sim.data_cleaning import clean_data
from cafe_sim.period_key import ALL_PERIOD_KEYS, period_key_to_display_name
from cafe_sim.configuration import create_experiment_definition, create_simulation_config
from cafe_sim.replication_runner import load_period_calibrations, run_experiment
from cafe_sim.validation_adequacy import check_validation_adequacy, generate_full_report
import yaml


def main():
    project_root = Path(__file__).parent.parent
    data_directory = project_root / "simulation_data"
    calibration_path = project_root / "config" / "period_calibrations.yaml"
    results_dir = project_root / "results"

    results_dir.mkdir(exist_ok=True)

    print("Validation Adequacy Assessment")
    print("=" * 80)

    print("\n1. Loading observation data...")
    raw_data = load_all_periods(data_directory)

    cleaned_data = {
        period_key: clean_data(raw_df, period_key)
        for period_key, raw_df in raw_data.items()
    }
    print(" Data loaded")

    print("\n2. Running validation experiments...")
    period_calibrations = load_period_calibrations(calibration_path)

    with open(calibration_path, "r") as f:
        calibrations = yaml.safe_load(f)

    validation_results = {}

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

    print(" Validation experiments complete")

    print("\n3. Performing adequacy checks...")

    adequacy_reports = {}

    for period_key in ALL_PERIOD_KEYS:
        report = check_validation_adequacy(
            period_key=period_key,
            simulated_results=validation_results[period_key],
            observed_data=cleaned_data[period_key],
            mean_wait_tolerance=0.5,
            ecdf_threshold=0.15,
            throughput_tolerance=0.05,
        )
        adequacy_reports[period_key] = report

        status = " PASS" if report['overall_adequate'] else " FAIL"
        print(f"   {period_key_to_display_name(period_key):20s}: {status}")

    print("\n4. Generating adequacy report...")
    full_report = generate_full_report(adequacy_reports)

    report_path = results_dir / "validation_adequacy_report.txt"
    with open(report_path, "w") as f:
        f.write(full_report)

    print(f" Report saved to: {report_path}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(full_report)


if __name__ == "__main__":
    main()
