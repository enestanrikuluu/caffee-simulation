import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cafe_sim.data_loading import DataLoader
from cafe_sim.data_cleaning import InputDataCleaner
from cafe_sim.period_key import PeriodKey
from cafe_sim.distribution_fitting import DistributionFitter
import numpy as np


def main():
    project_root = Path(__file__).parent.parent
    data_directory = project_root / "simulation_data"
    results_dir = project_root / "results"
    
    results_dir.mkdir(exist_ok=True)
    
    print("Distribution Fitting Analysis")
    print("=" * 100)
    
    print("\n1. Loading and cleaning observation data...")
    loader = DataLoader(data_directory)
    raw_data = loader.load_all_periods()
    
    cleaner = InputDataCleaner()
    cleaned_data = {
        period_key: cleaner.clean(raw_df, period_key)
        for period_key, raw_df in raw_data.items()
    }
    print("✓ Data loaded and cleaned")
    
    print("\n2. Fitting distributions to service times and interarrival times...\n")
    
    fitter = DistributionFitter(num_bins_chi_square=10, min_expected_frequency=5.0)
    
    all_reports = {}
    
    for period_key in PeriodKey:
        df = cleaned_data[period_key]
        period_name = period_key.to_display_name()
        
        print(f"   Analyzing {period_name}...")
        
        service_times = df["service_time_min"].dropna().values
        if len(service_times) >= 30:
            service_report = fitter.fit_all_distributions(
                service_times, 
                variable_name=f"{period_name} - Service Time"
            )
            all_reports[f"{period_key.value}_service"] = service_report
            print(f"      Service times: {service_report.recommended_distribution.value} "
                  f"(n={service_report.sample_size}, AIC rank #1)")
        
        interarrival_times = df["interarrival_min"].dropna().values
        interarrival_times = interarrival_times[interarrival_times > 0]
        if len(interarrival_times) >= 30:
            interarrival_report = fitter.fit_all_distributions(
                interarrival_times,
                variable_name=f"{period_name} - Interarrival Time"
            )
            all_reports[f"{period_key.value}_interarrival"] = interarrival_report
            print(f"      Interarrival times: {interarrival_report.recommended_distribution.value} "
                  f"(n={interarrival_report.sample_size}, AIC rank #1)")
        
        drink_services = df[df["service_time_min"] <= 1.0]["service_time_min"].dropna().values
        if len(drink_services) >= 30:
            drink_report = fitter.fit_all_distributions(
                drink_services,
                variable_name=f"{period_name} - Drink Service Time"
            )
            all_reports[f"{period_key.value}_drink_service"] = drink_report
            print(f"      Drink services: {drink_report.recommended_distribution.value} "
                  f"(n={drink_report.sample_size})")
        
        food_services = df[df["service_time_min"] > 1.0]["service_time_min"].dropna().values
        if len(food_services) >= 30:
            food_report = fitter.fit_all_distributions(
                food_services,
                variable_name=f"{period_name} - Food Service Time"
            )
            all_reports[f"{period_key.value}_food_service"] = food_report
            print(f"      Food services: {food_report.recommended_distribution.value} "
                  f"(n={food_report.sample_size})")
    
    print("\n3. Generating comprehensive report...")
    
    report_text = fitter.generate_text_report(all_reports)
    
    report_path = results_dir / "distribution_fitting_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)
    
    print(f"✓ Report saved to: {report_path}")
    
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    
    print("\nRecommended Distributions by Variable Type:\n")
    
    service_times_recs = {}
    interarrival_recs = {}
    drink_recs = {}
    food_recs = {}
    
    for key, report in all_reports.items():
        dist_name = report.recommended_distribution.value
        if "interarrival" in key:
            interarrival_recs[report.variable_name] = dist_name
        elif "drink_service" in key:
            drink_recs[report.variable_name] = dist_name
        elif "food_service" in key:
            food_recs[report.variable_name] = dist_name
        elif "service" in key:
            service_times_recs[report.variable_name] = dist_name
    
    if service_times_recs:
        print("Overall Service Times:")
        for var, dist in service_times_recs.items():
            print(f"  • {var}: {dist}")
    
    if drink_recs:
        print("\nDrink Service Times (≤1.0 min):")
        for var, dist in drink_recs.items():
            print(f"  • {var}: {dist}")
    
    if food_recs:
        print("\nFood Service Times (>1.0 min):")
        for var, dist in food_recs.items():
            print(f"  • {var}: {dist}")
    
    if interarrival_recs:
        print("\nInterarrival Times:")
        for var, dist in interarrival_recs.items():
            print(f"  • {var}: {dist}")
    
    print("\n" + "=" * 100)
    print("NOTE: While theoretical distributions are fitted, the simulation currently uses")
    print("empirical distributions (inverse-CDF sampling). Theoretical fits provide insight")
    print("into data characteristics but may lose important tail behavior or multimodality.")
    print("=" * 100)


if __name__ == "__main__":
    main()
