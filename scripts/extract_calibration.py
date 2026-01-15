import yaml
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from cafe_sim.data_loading import DataLoader
from cafe_sim.data_cleaning import InputDataCleaner
from cafe_sim.calibration_extractor import CalibrationExtractor, ThresholdDiagnosticAnalyzer
from cafe_sim.period_key import PeriodKey


def main():
    project_root = Path(__file__).parent.parent
    data_directory = project_root / "simulation_data"
    config_directory = project_root / "config"
    results_directory = project_root / "results"
    
    config_directory.mkdir(exist_ok=True)
    results_directory.mkdir(exist_ok=True)
    
    print("Loading observation data from Excel files...")
    loader = DataLoader(data_directory)
    raw_data = loader.load_all_periods()
    
    print("Cleaning and normalizing data...")
    cleaner = InputDataCleaner()
    cleaned_data = {
        period_key: cleaner.clean(raw_df, period_key)
        for period_key, raw_df in raw_data.items()
    }
    
    print("\nGenerating calibration statistics...")
    threshold_analyzer = ThresholdDiagnosticAnalyzer()
    extractor = CalibrationExtractor(threshold_analyzer)
    
    baseline_threshold = 1.0
    period_statistics = extractor.extract_period_statistics(
        cleaned_data, baseline_threshold=baseline_threshold
    )
    
    print("\nRunning threshold diagnostic analysis...")
    diagnostics = extractor.generate_threshold_diagnostics(cleaned_data)
    
    print("\nPeriod Statistics Summary:")
    print("=" * 80)
    for period_key in PeriodKey:
        stats = period_statistics[period_key]
        print(f"\n{period_key.to_display_name()}:")
        print(f"  Observations: {stats['observation_count']}")
        
        run_length = stats.get('run_length_min', None)
        if run_length is not None:
            print(f"  Run length: {run_length:.1f} min")
        else:
            print(f"  Run length: N/A")
        
        arrival_rate = stats.get('arrival_rate_per_hour', None)
        if arrival_rate is not None:
            print(f"  Arrival rate: {arrival_rate:.1f} customers/hr")
        else:
            print(f"  Arrival rate: N/A")
        
        lambda_val = stats.get('poisson_lambda_per_minute', None)
        if lambda_val is not None:
            print(f"  Poisson λ: {lambda_val:.3f} per min")
        else:
            print(f"  Poisson λ: N/A")
        
        print(f"  P(Drink): {stats['p_drink']:.2f}")
        print(f"  Mean service (Drink): {stats['mean_service_drink']:.2f} min")
        print(f"  Mean service (Food): {stats['mean_service_food']:.2f} min")
        if "mean_wait_min" in stats:
            print(f"  Mean wait: {stats['mean_wait_min']:.2f} min")
            print(f"  P(wait > 0): {stats['p_wait_positive']:.2f}")
    
    calibration_config = {}
    for period_key in PeriodKey:
        stats = period_statistics[period_key]
        calibration_config[period_key.value] = {
            "run_length_min": stats.get("run_length_min", 60.0),
            "suggested_barista_count": _suggest_barista_count(period_key),
            "arrival_rate_per_hour": stats.get("arrival_rate_per_hour", 50.0),
            "poisson_lambda_per_minute": stats.get("poisson_lambda_per_minute", 1.0),
            "p_drink": stats["p_drink"],
            "empirical_interarrivals_min": stats.get("empirical_interarrivals_min", [1.0]),
            "empirical_service_drink_min": stats["empirical_service_drink_min"],
            "empirical_service_food_min": stats["empirical_service_food_min"],
        }
    
    calibration_path = config_directory / "period_calibrations.yaml"
    with open(calibration_path, "w") as f:
        yaml.dump(calibration_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✓ Calibration configuration saved to: {calibration_path}")
    
    diagnostic_report_path = results_directory / "calibration_diagnostics.txt"
    with open(diagnostic_report_path, "w") as f:
        f.write("Threshold Diagnostic Analysis Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Baseline threshold: {baseline_threshold} minute(s)\n\n")
        
        for period_key in PeriodKey:
            if period_key not in diagnostics:
                continue
            
            f.write(f"\n{period_key.to_display_name()}\n")
            f.write("-" * 40 + "\n\n")
            
            f.write("Fixed threshold splits:\n")
            for diag in diagnostics[period_key]["threshold_splits"]:
                f.write(f"  Threshold: {diag.threshold:.1f} min\n")
                f.write(f"    P(Drink): {diag.p_drink:.3f}\n")
                f.write(f"    Mean Drink: {diag.mean_drink:.2f} min\n")
                f.write(f"    Mean Food: {diag.mean_food:.2f} min\n")
                f.write(f"    Separation: {diag.mean_difference:.2f} min\n")
                f.write(f"    Overlap coeff: {diag.overlap_coefficient:.3f}\n\n")
            
            kmeans_diag = diagnostics[period_key]["kmeans_split"]
            if kmeans_diag:
                f.write("K-means clustering (k=2):\n")
                f.write(f"  Inferred threshold: {kmeans_diag.threshold:.2f} min\n")
                f.write(f"  P(Drink): {kmeans_diag.p_drink:.3f}\n")
                f.write(f"  Mean Drink: {kmeans_diag.mean_drink:.2f} min\n")
                f.write(f"  Mean Food: {kmeans_diag.mean_food:.2f} min\n")
                f.write(f"  Separation: {kmeans_diag.mean_difference:.2f} min\n")
                f.write(f"  Overlap coeff: {kmeans_diag.overlap_coefficient:.3f}\n\n")
    
    print(f"✓ Diagnostic report saved to: {diagnostic_report_path}")
    print("\nCalibration extraction complete!")


def _suggest_barista_count(period_key: PeriodKey) -> int:
    suggestions = {
        PeriodKey.MONDAY_MORNING: 3,
        PeriodKey.MONDAY_AFTERNOON: 4,
        PeriodKey.MONDAY_EVENING: 2,
        PeriodKey.TUESDAY_MORNING: 1,
        PeriodKey.TUESDAY_AFTERNOON: 2,
        PeriodKey.TUESDAY_EVENING: 2,
    }
    return suggestions[period_key]


if __name__ == "__main__":
    main()
