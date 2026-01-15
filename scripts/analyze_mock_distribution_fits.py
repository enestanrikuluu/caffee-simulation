import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cafe_sim.distribution_fitting import DistributionFitter
import pandas as pd
import numpy as np


def main():
    project_root = Path(__file__).parent.parent
    mock_data_dir = project_root / "simulation_data" / "mock"
    results_dir = project_root / "results"
    
    results_dir.mkdir(exist_ok=True)
    
    print("Distribution Fitting Analysis - Mock Data (Continuous Time)")
    print("=" * 100)
    
    csv_files = sorted(mock_data_dir.glob("*_mock.csv"))
    
    if not csv_files:
        print(f"ERROR: No mock data files found in {mock_data_dir}")
        print("Run 'python scripts/generate_mock_data.py' first.")
        return
    
    print(f"\n1. Loading {len(csv_files)} mock data files...")
    
    mock_data = {}
    for file_path in csv_files:
        period_name = file_path.stem.replace("_mock", "")
        df = pd.read_csv(file_path)
        mock_data[period_name] = df
        print(f"   ✓ {period_name}: {len(df)} customers")
    
    print("\n2. Fitting distributions to continuous-time data...\n")
    
    fitter = DistributionFitter(num_bins_chi_square=12, min_expected_frequency=5.0)
    
    all_reports = {}
    
    for period_name, df in mock_data.items():
        print(f"   Analyzing {period_name}...")
        
        service_times_sec = df["service_time_sec"].dropna().values
        if len(service_times_sec) >= 30:
            service_report = fitter.fit_all_distributions(
                service_times_sec, 
                variable_name=f"{period_name} - Service Time (sec)"
            )
            all_reports[f"{period_name}_service"] = service_report
            
            best_dist = service_report.recommended_distribution.value
            best_result = service_report.fit_results[0]
            ks_status = "PASS" if best_result.ks_p_value > 0.05 else "FAIL"
            print(f"      Service times: {best_dist} (K-S p={best_result.ks_p_value:.4f} {ks_status})")
        
        interarrival_times_sec = df["interarrival_sec"].dropna().values
        interarrival_times_sec = interarrival_times_sec[interarrival_times_sec > 0]
        if len(interarrival_times_sec) >= 30:
            interarrival_report = fitter.fit_all_distributions(
                interarrival_times_sec,
                variable_name=f"{period_name} - Interarrival Time (sec)"
            )
            all_reports[f"{period_name}_interarrival"] = interarrival_report
            
            best_dist = interarrival_report.recommended_distribution.value
            best_result = interarrival_report.fit_results[0]
            ks_status = "PASS" if best_result.ks_p_value > 0.05 else "FAIL"
            print(f"      Interarrival times: {best_dist} (K-S p={best_result.ks_p_value:.4f} {ks_status})")
        
        drink_services_sec = df[df["service_type"] == "drink"]["service_time_sec"].dropna().values
        if len(drink_services_sec) >= 30:
            drink_report = fitter.fit_all_distributions(
                drink_services_sec,
                variable_name=f"{period_name} - Drink Service Time (sec)"
            )
            all_reports[f"{period_name}_drink_service"] = drink_report
            
            best_dist = drink_report.recommended_distribution.value
            best_result = drink_report.fit_results[0]
            ks_status = "PASS" if best_result.ks_p_value > 0.05 else "FAIL"
            print(f"      Drink services: {best_dist} (K-S p={best_result.ks_p_value:.4f} {ks_status})")
        
        food_services_sec = df[df["service_type"] == "food"]["service_time_sec"].dropna().values
        if len(food_services_sec) >= 30:
            food_report = fitter.fit_all_distributions(
                food_services_sec,
                variable_name=f"{period_name} - Food Service Time (sec)"
            )
            all_reports[f"{period_name}_food_service"] = food_report
            
            best_dist = food_report.recommended_distribution.value
            best_result = food_report.fit_results[0]
            ks_status = "PASS" if best_result.ks_p_value > 0.05 else "FAIL"
            print(f"      Food services: {best_dist} (K-S p={best_result.ks_p_value:.4f} {ks_status})")
    
    print("\n3. Generating comprehensive report...")
    
    report_text = fitter.generate_text_report(all_reports)
    
    report_path = results_dir / "mock_distribution_fitting_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)
    
    print(f"✓ Report saved to: {report_path}")
    
    print("\n" + "=" * 100)
    print("GOODNESS-OF-FIT SUMMARY (K-S Test at α=0.05)")
    print("=" * 100)
    
    passed_tests = 0
    total_tests = 0
    
    for var_name, report in all_reports.items():
        best_result = report.fit_results[0]
        total_tests += 1
        
        if not np.isnan(best_result.ks_p_value) and best_result.ks_p_value > 0.05:
            passed_tests += 1
            status = "✓ PASS"
        else:
            status = "✗ FAIL"
        
        dist_name = best_result.distribution.value
        p_val = best_result.ks_p_value if not np.isnan(best_result.ks_p_value) else float('nan')
        
        print(f"{status}  {report.variable_name:60s}  {dist_name:12s}  p={p_val:.4f}")
    
    print("\n" + "=" * 100)
    print(f"OVERALL: {passed_tests}/{total_tests} distributions pass K-S test ({passed_tests/total_tests*100:.1f}%)")
    print("=" * 100)
    
    if passed_tests > total_tests * 0.5:
        print("\n✓ GOOD NEWS: With continuous-time data, theoretical distributions fit much better!")
        print("  This confirms the original data issue was discretization, not fundamental unfitness.")
    else:
        print("\n⚠ Even with continuous data, many distributions fail fit tests.")
        print("  This suggests inherent complexity (mixtures, non-stationarity) in café behavior.")


if __name__ == "__main__":
    main()
