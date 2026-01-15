import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cafe_sim.data_loading import DataLoader
from cafe_sim.data_cleaning import InputDataCleaner
from cafe_sim.period_key import PeriodKey
from cafe_sim.configuration import ExperimentDefinition, SimulationConfig
from cafe_sim.replication_runner import ReplicationRunner
from cafe_sim.validation_adequacy import ValidationAdequacyChecker
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
    loader = DataLoader(data_directory)
    raw_data = loader.load_all_periods()
    
    cleaner = InputDataCleaner()
    cleaned_data = {
        period_key: cleaner.clean(raw_df, period_key)
        for period_key, raw_df in raw_data.items()
    }
    print("✓ Data loaded")
    
    print("\n2. Running validation experiments...")
    runner = ReplicationRunner(calibration_path)
    
    with open(calibration_path, "r") as f:
        calibrations = yaml.safe_load(f)
    
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
                arrival_model_type="poisson",
            ),
            rng_seed=42,
        )
        
        results = runner.run_experiment(experiment)
        validation_results[period_key] = results
    
    print("✓ Validation experiments complete")
    
    print("\n3. Performing adequacy checks...")
    checker = ValidationAdequacyChecker(
        mean_wait_tolerance_min=0.5,
        ecdf_max_deviation_threshold=0.15,
        throughput_tolerance_pct=0.05,
    )
    
    adequacy_reports = {}
    
    for period_key in PeriodKey:
        report = checker.check_validation_adequacy(
            period_key=period_key,
            simulated_results=validation_results[period_key],
            observed_data=cleaned_data[period_key],
        )
        adequacy_reports[period_key] = report
        
        status = "✓ PASS" if report.overall_adequate else "✗ FAIL"
        print(f"   {period_key.to_display_name():20s}: {status}")
    
    print("\n4. Generating adequacy report...")
    full_report = checker.generate_full_report(adequacy_reports)
    
    report_path = results_dir / "validation_adequacy_report.txt"
    with open(report_path, "w") as f:
        f.write(full_report)
    
    print(f"✓ Report saved to: {report_path}")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(full_report)


if __name__ == "__main__":
    main()
