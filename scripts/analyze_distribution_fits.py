import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cafe_sim.data_loading import load_all_periods
from cafe_sim.data_cleaning import clean_data
from cafe_sim.period_key import ALL_PERIOD_KEYS, period_key_to_display_name
from cafe_sim.distribution_fitting import fit_all_distributions, generate_text_report
import numpy as np


def main():
    project_root = Path(__file__).parent.parent
    data_directory = project_root / "simulation_data"
    results_dir = project_root / "results"

    results_dir.mkdir(exist_ok=True)

    print("Distribution Fitting Analysis")
    print("=" * 100)

    print("\n1. Loading and cleaning observation data...")
    raw_data = load_all_periods(data_directory)

    cleaned_data = {
        period_key: clean_data(raw_df, period_key)
        for period_key, raw_df in raw_data.items()
    }
    print(" Data loaded and cleaned")

    print("\n2. Fitting distributions to service times and interarrival times...\n")

    all_reports = {}

    for period_key in ALL_PERIOD_KEYS:
        df = cleaned_data[period_key]
        period_name = period_key_to_display_name(period_key)

        print(f"   Analyzing {period_name}...")

        service_times = df["service_time_min"].dropna().values
        if len(service_times) >= 30:
            service_report = fit_all_distributions(
                service_times,
                variable_name=f"{period_name} - Service Time"
            )
            all_reports[f"{period_key}_service"] = service_report
            print(f"      Service times: {service_report['recommended_distribution']} "
                  f"(n={service_report['sample_size']}, AIC rank #1)")

        interarrival_times = df["interarrival_min"].dropna().values
        interarrival_times = interarrival_times[interarrival_times > 0]
        if len(interarrival_times) >= 30:
            interarrival_report = fit_all_distributions(
                interarrival_times,
                variable_name=f"{period_name} - Interarrival Time"
            )
            all_reports[f"{period_key}_interarrival"] = interarrival_report
            print(f"      Interarrival times: {interarrival_report['recommended_distribution']} "
                  f"(n={interarrival_report['sample_size']}, AIC rank #1)")

        # Use service_type column if available, otherwise fall back to threshold
        if "service_type" in df.columns:
            drink_services = df[df["service_type"] == "drink"]["service_time_min"].dropna().values
            food_services = df[df["service_type"] == "food"]["service_time_min"].dropna().values
        else:
            drink_services = df[df["service_time_min"] <= 1.0]["service_time_min"].dropna().values
            food_services = df[df["service_time_min"] > 1.0]["service_time_min"].dropna().values

        if len(drink_services) >= 30:
            drink_report = fit_all_distributions(
                drink_services,
                variable_name=f"{period_name} - Drink Service Time"
            )
            all_reports[f"{period_key}_drink_service"] = drink_report
            print(f"      Drink services: {drink_report['recommended_distribution']} "
                  f"(n={drink_report['sample_size']})")

        if len(food_services) >= 30:
            food_report = fit_all_distributions(
                food_services,
                variable_name=f"{period_name} - Food Service Time"
            )
            all_reports[f"{period_key}_food_service"] = food_report
            print(f"      Food services: {food_report['recommended_distribution']} "
                  f"(n={food_report['sample_size']})")

    print("\n3. Generating comprehensive report...")

    report_text = generate_text_report(all_reports)

    report_path = results_dir / "distribution_fitting_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)

    print(f" Report saved to: {report_path}")

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    print("\nRecommended Distributions by Variable Type:\n")

    service_times_recs = {}
    interarrival_recs = {}
    drink_recs = {}
    food_recs = {}

    for key, report in all_reports.items():
        dist_name = report['recommended_distribution']
        if "interarrival" in key:
            interarrival_recs[report['variable_name']] = dist_name
        elif "drink_service" in key:
            drink_recs[report['variable_name']] = dist_name
        elif "food_service" in key:
            food_recs[report['variable_name']] = dist_name
        elif "service" in key:
            service_times_recs[report['variable_name']] = dist_name

    if service_times_recs:
        print("Overall Service Times:")
        for var, dist in service_times_recs.items():
            print(f"  - {var}: {dist}")

    if drink_recs:
        print("\nDrink Service Times (from service_type labels):")
        for var, dist in drink_recs.items():
            print(f"  - {var}: {dist}")

    if food_recs:
        print("\nFood Service Times (from service_type labels):")
        for var, dist in food_recs.items():
            print(f"  - {var}: {dist}")

    if interarrival_recs:
        print("\nInterarrival Times:")
        for var, dist in interarrival_recs.items():
            print(f"  - {var}: {dist}")

    print("\n" + "=" * 100)
    print("NOTE: While theoretical distributions are fitted, the simulation currently uses")
    print("empirical distributions (inverse-CDF sampling). Theoretical fits provide insight")
    print("into data characteristics but may lose important tail behavior or multimodality.")
    print("=" * 100)


if __name__ == "__main__":
    main()
