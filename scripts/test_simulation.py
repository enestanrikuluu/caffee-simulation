import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cafe_sim.period_key import TUESDAY_MORNING, period_key_to_display_name
from cafe_sim.configuration import create_experiment_definition, create_simulation_config
from cafe_sim.replication_runner import load_period_calibrations, run_experiment


def main():
    project_root = Path(__file__).parent.parent
    calibration_path = project_root / "config" / "period_calibrations.yaml"

    print("Running simple validation test...")
    print("=" * 80)

    period_calibrations = load_period_calibrations(calibration_path)

    test_period = TUESDAY_MORNING

    experiment = create_experiment_definition(
        name="test_tuesday_morning",
        period_key=test_period,
        barista_count=1,
        run_length_min=95.0,
        replication_count=5,
        simulation_config=create_simulation_config(
            warmup_min=0.0,
            initial_queue_size=0,
            initial_busy_baristas=0,
            arrival_model_type="poisson",
        ),
        rng_seed=42,
    )

    print(f"\nRunning experiment: {experiment['name']}")
    print(f"Period: {period_key_to_display_name(test_period)}")
    print(f"Baristas: {experiment['barista_count']}")
    print(f"Run length: {experiment['run_length_min']} min")
    print(f"Replications: {experiment['replication_count']}")

    results = run_experiment(period_calibrations, experiment)

    print("\n" + "=" * 80)
    print("Results (Full Period):")
    print("=" * 80)

    for kpi_name, ci in results['full_period'].items():
        print(f"{kpi_name:30s}: {ci['mean']:8.3f} +/- {ci['upper'] - ci['mean']:6.3f}  "
              f"[{ci['lower']:7.3f}, {ci['upper']:7.3f}]")

    print("\n Simulation engine test completed successfully!")


if __name__ == "__main__":
    main()
