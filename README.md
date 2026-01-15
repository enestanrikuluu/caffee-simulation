# Café Discrete-Event Simulation

A production-quality discrete-event simulation system for modeling café operations across 6 observation periods (Monday/Tuesday × Morning/Afternoon/Evening). Implements Banks event-scheduling architecture with SOLID principles, comprehensive input modeling, statistical output analysis, and experiment infrastructure.

## System Overview

**Objective**: Quantify congestion and performance; test staffing and service-mix scenarios for a café operation.

**Approach**: Event-scheduling simulation with:

- Multi-server queueing (m parallel baristas, single FCFS queue)
- Dual arrival models: Poisson-per-minute (handles discretized data) + empirical interarrivals
- Service-type mixture: Drink vs. Drink+Food with period-specific probabilities and conditional empirical distributions
- Configurable warm-up periods and initial conditions
- Replication-based output analysis with confidence intervals

## Project Structure

```
cafe_sim/                           # Core simulation package
├── period_key.py                   # Period enumeration (Mon/Tue × Morning/Afternoon/Evening)
├── service_type.py                 # Service type enumeration (Drink, Drink+Food)
├── configuration.py                # Dataclasses: SimulationConfig, ExperimentDefinition, KPISet
├── data_loading.py                 # DataLoader: reads Excel observation files
├── data_cleaning.py                # InputDataCleaner: normalizes columns, converts times, validates
├── calibration_extractor.py        # Extracts λ, p_drink, empirical CDFs from observations
├── empirical_distribution.py       # Inverse-CDF sampling from empirical data
├── arrival_process_model.py        # Poisson-per-minute + empirical arrival models with factory
├── service_type_model.py           # Service-type mixture with conditional distributions
├── event_definitions.py            # Event types and ordering
├── future_event_list.py            # Priority queue for events
├── simulation_state.py             # State: queues, barista availability, time-weighted areas
├── simulation_engine.py            # CafeSimulationEngine: Banks event-scheduling logic
├── kpi_aggregation.py              # Computes confidence intervals across replications
├── replication_runner.py           # Runs R replications, aggregates KPIs
└── validation_adequacy.py          # Automated validation with formal thresholds└── distribution_fitting.py         # Theoretical distribution fitting (exponential, gamma, Weibull, lognormal, normal, uniform) with goodness-of-fit tests (chi-square, K-S, Anderson-Darling)
scripts/
├── extract_calibration.py          # Generates config/period_calibrations.yaml
├── test_simulation.py              # Simple validation test
├── run_experiments.py              # Full experiment suite
├── run_validation_adequacy.py      # Formal adequacy assessment
├── analyze_distribution_fits.py    # Distribution fitting with goodness-of-fit tests
└── generate_visualizations.py      # Generate 8 publication-quality plots

tests/
├── test_empirical_sampling.py      # Unit tests for empirical distribution (10 tests)
├── test_event_ordering.py          # Unit tests for FutureEventList (5 tests)
└── test_area_integrals.py          # Unit tests for time-weighted statistics (10 tests)

config/
└── period_calibrations.yaml        # Period-specific λ, p_drink, empirical CDFs

results/
├── calibration_diagnostics.txt     # Threshold analysis (1.0/1.5/2.0 min, k-means)
├── experiment_summary.txt          # KPI results from experiments
├── distribution_fitting_report.txt # Theoretical distribution fits with test statistics
├── validation_adequacy_report.txt  # Pass/fail assessment for validation
└── plots/                          # Generated visualizations
    ├── calibration_service_histograms.png
    ├── calibration_interarrival_ecdfs.png
    ├── threshold_diagnostics.png
    ├── validation_kpi_comparison.png
    ├── validation_wait_ecdf.png
    ├── staffing_sweep_monday_afternoon.png
    ├── service_mix_tuesday_morning.png
    └── warmup_analysis_monday_afternoon.png
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Extract calibration from observation data
python scripts/extract_calibration.py

# Run unit tests
pytest tests/ -v
```

## Usage

### 1. Calibration Extraction

Analyzes observation files in `simulation_data/` and generates `config/period_calibrations.yaml`:

```bash
python scripts/extract_calibration.py
```

**Outputs**:

- `config/period_calibrations.yaml`: Poisson λ, p_drink, empirical service CDFs per period
- `results/calibration_diagnostics.txt`: Threshold sensitivity (1.0/1.5/2.0 min, k-means)

### 2. Distribution Fitting Analysis

Fits theoretical distributions (exponential, gamma, Weibull, lognormal, normal, uniform) to service and interarrival times with goodness-of-fit tests:

```bash
python scripts/analyze_distribution_fits.py
```

**Tests performed**:

- **Chi-Square Test**: Compares observed vs. expected frequencies (p > 0.05 = good fit)
- **Kolmogorov-Smirnov Test**: Maximum ECDF deviation (p > 0.05 = good fit)
- **Anderson-Darling Test**: Weighted ECDF comparison (statistic < critical value = good fit)
- **AIC/BIC**: Information criteria for model selection (lower = better)

**Output**: `results/distribution_fitting_report.txt` with ranked distributions per variable

**Key findings**:

- Service times: Mixed (gamma, lognormal, uniform depending on period)
- Interarrival times: Predominantly lognormal
- Drink services: Lognormal (≤1.0 min)
- Food services: Mixed (lognormal, uniform)

**Note**: Simulation uses empirical distributions to preserve tail behavior and multimodality.

### 3. Validation Adequacy Assessment

```bash
python scripts/run_validation_adequacy.py
```

**Automated adequacy tests** comparing simulated vs. observed metrics:

- **Mean Agreement**: |simulated - observed| ≤ 0.5 min for mean wait time
- **ECDF Deviation**: Kolmogorov-Smirnov statistic ≤ 0.15 for wait time distributions
- **Throughput Match**: Relative difference ≤ 5% for customer throughput

**Output**: `results/validation_adequacy_report.txt` with pass/fail per period.

### 4. Run Experiments

```bash
python scripts/run_experiments.py
```

**Experiments**:

1. **Validation**: Simulate 6 periods with observed barista counts (30 replications)
2. **Staffing Sweep**: Monday afternoon with m=3,4,5 baristas
3. **Service-Mix Sensitivity**: Vary p_drink by ±15% (Tuesday morning)
4. **Warm-up Sensitivity**: Monday afternoon with 0/15/30 min warm-up

**Output**: `results/experiment_summary.txt` with KPI means and 95% CIs.

### 5. Generate Visualizations

```bash
python scripts/generate_visualizations.py
```

**Generates 8 plots in** `results/plots/`:

1. **calibration_service_histograms.png**: Service time distributions split by DRINK/FOOD threshold per period
2. **calibration_interarrival_ecdfs.png**: Empirical CDFs of interarrival times per period
3. **threshold_diagnostics.png**: Separation quality (mean difference vs. overlap) across thresholds
4. **validation_kpi_comparison.png**: Simulated vs. observed KPIs (wait time, utilization, throughput)
5. **validation_wait_ecdf.png**: Wait time ECDF comparison (placeholder for replication traces)
6. **staffing_sweep_monday_afternoon.png**: KPIs across m=3,4,5 baristas with confidence intervals
7. **service_mix_tuesday_morning.png**: Impact of p_drink perturbation on wait time and utilization
8. **warmup_analysis_monday_afternoon.png**: Full-period vs. post-warmup KPI comparison

### 6. Custom Experiments

```python
from pathlib import Path
from cafe_sim.period_key import PeriodKey
from cafe_sim.configuration import ExperimentDefinition, SimulationConfig
from cafe_sim.replication_runner import ReplicationRunner

runner = ReplicationRunner(Path("config/period_calibrations.yaml"))

experiment = ExperimentDefinition(
    name="custom_experiment",
    period_key=PeriodKey.MONDAY_AFTERNOON,
    barista_count=4,
    run_length_min=100.0,
    replication_count=50,
    simulation_config=SimulationConfig(
        warmup_min=15.0,
        initial_queue_size=0,
        initial_busy_baristas=0,
        arrival_model_type="poisson",  # or "empirical"
    ),
    p_drink_perturbation=0.0,  # ±delta for service-mix scenarios
    rng_seed=42,
)

results = runner.run_experiment(experiment)
print(f"Mean wait: {results.full_period['mean_wait_min'].mean:.2f} min")
```

## Key Design Decisions

### Arrival Modeling (Poisson-per-minute as baseline)

**Problem**: Observations recorded at 1-minute resolution → many zero interarrivals (measurement artifact).

**Solution**: Model as Poisson count per minute, place arrivals uniformly within minute.

- Handles simultaneous arrivals naturally
- Avoids illegal negative times
- Transparent calibration: λ = total_arrivals / run_length_min

**Alternative**: Empirical interarrivals + U(0,1) jitter (available via `arrival_model_type="empirical"`).

### Service-Type Mixture (1.0-minute threshold)

**Problem**: No service-type labels in data.

**Solution**: Period-specific p_drink + conditional empirical distributions.

- Threshold: service ≤ 1.0 min → DRINK; > 1.0 min → DRINK+FOOD
- Transparent assumption, easy sensitivity testing
- Diagnostic analysis (1.0/1.5/2.0 thresholds, k-means) in calibration report

**Justification**: p_drink ranges from 0.25 (Monday morning, food-heavy) to 0.89 (Tuesday morning, drink-heavy), aligns with business intuition.

### Warm-up Handling (terminating system)

**Default**: No warm-up (empty-and-idle start) for validation against observed data.

**Scenarios**: Configurable warm-up (e.g., 15 min) with dual KPI output:

- **Full-period**: Includes transient (matches validation)
- **Post-warm-up**: Measurement phase only (sensitivity experiments)

### Validation Approach (descriptive, not inferential)

**Checks**:

- Simulated mean within observed variability
- Distribution shape agreement (ECDF comparison)
- Throughput match

**Rationale**: Observed n=1 realization per period; formal hypothesis testing overstates precision. Focus on effect sizes and practical adequacy.

## KPIs Tracked

- \*Visualization Gallery

All plots are generated in [results/plots/](results/plots/):

### Calibration Analysis

- **Service Histograms**: Shows DRINK (≤1 min) vs. DRINK+FOOD (>1 min) distributions per period
- **Interarrival ECDFs**: Empirical cumulative distributions showing discretization at 1-minute intervals
- **Threshold Diagnostics**: Separation quality across 1.0/1.5/2.0 min thresholds (mean difference vs. overlap)

### Validation

- **KPI Comparison**: Simulated means with 95% CIs vs. observed values (wait time, utilization, throughput)
- **Wait Time ECDFs**: Distribution shape comparison (requires replication-level traces for full implementation)

### Scenario Analysis

- **Staffing Sweep**: Monday afternoon performance across m=3,4,5 baristas showing diminishing returns
- **Service Mix Sensitivity**: Tuesday morning wait times increase dramatically as p_drink decreases (more food orders)
- **Warm-up Analysis**: Full-period vs. post-warmup KPIs showing transient effects

## Future Enhancements

1. **NHPP**: Time-varying λ(t) in 5-10 min buckets (if validation shows timing bias)
2. **Validation adequacy checker**: Automated agreement tests with thresholds
3. **Sequential stopping**: Continue replications until CI half-width < target
4. **Queue time-series**: Event-based step plots of queue length over simulation runs

| Period            | n   | λ (per min) | P(Drink) | Suggested m |
| ----------------- | --- | ----------- | -------- | ----------- |
| Monday Morning    | 105 | 1.020       | 0.25     | 3           |
| Monday Afternoon  | 99  | 2.000       | 0.61     | 4           |
| Monday Evening    | 53  | 0.815       | 0.53     | 2           |
| Tuesday Morning   | 83  | 0.874       | 0.87     | 1           |
| Tuesday Afternoon | 87  | 1.192       | 0.72     | 2           |
| Tuesday Evening   | 58  | 0.817       | 0.36     | 2           |

## Future Enhancements

1. **NHPP**: Time-varying λ(t) in 5-10 min buckets (if validation shows timing bias)
2. **Visualization**: Calibration plots, validation ECDFs, scenario comparisons
3. **Validation adequacy checker**: Automated agreement tests with thresholds
4. **Sequential stopping**: Continue replications until CI half-width < target

## Testing

```bash
# Run all unit tests (25 tests total)
pytest tests/ -v

# Run specific test suite
pytest tests/test_empirical_sampling.py -v  # 10 tests: inverse-CDF sampling correctness
pytest tests/test_event_ordering.py -v      # 5 tests: priority queue ordering
pytest tests/test_area_integrals.py -v      # 10 tests: time-weighted area accumulation

# Quick validation test
python scripts/test_simulation.py
```

## References

- Banks, J. et al. (2010). _Discrete-Event System Simulation_ (5th ed.). Pearson.
- Law, A. M. (2015). _Simulation Modeling and Analysis_ (5th ed.). McGraw-Hill.
