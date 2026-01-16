import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from cafe_sim.period_key import ALL_PERIOD_KEYS, period_key_to_display_name


def setup_visualization_style():
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 10


def ensure_output_dir(output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def plot_calibration_service_histograms(cleaned_data, output_dir, threshold=1.0):
    setup_visualization_style()
    output_path = ensure_output_dir(output_dir)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Service Time Distributions by Period (Probabilistic Labels)',
                 fontsize=14, fontweight='bold')

    for idx, period_key in enumerate(ALL_PERIOD_KEYS):
        ax = axes[idx // 3, idx % 3]
        df = cleaned_data[period_key]

        if 'service_time_min' not in df.columns or len(df) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(period_key_to_display_name(period_key))
            continue

        service_times = df['service_time_min'].dropna()

        # Use service_type column if available, otherwise fall back to threshold
        if 'service_type' in df.columns:
            drink_mask = df['service_type'] == 'drink'
            food_mask = df['service_type'] == 'food'
            drink_times = df.loc[drink_mask, 'service_time_min'].dropna()
            food_times = df.loc[food_mask, 'service_time_min'].dropna()
        else:
            drink_times = service_times[service_times <= threshold]
            food_times = service_times[service_times > threshold]

        bins = np.arange(0, max(service_times.max() + 1, 6), 0.5)

        ax.hist(drink_times, bins=bins, alpha=0.7, label='DRINK', color='skyblue', edgecolor='black')
        ax.hist(food_times, bins=bins, alpha=0.7, label='DRINK+FOOD', color='salmon', edgecolor='black')

        total = len(drink_times) + len(food_times)
        p_drink = len(drink_times) / total if total > 0 else 0

        ax.set_xlabel('Service Time (min)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{period_key_to_display_name(period_key)}\nP(Drink)={p_drink:.2f}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    file_path = output_path / 'calibration_service_histograms.png'
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()
    return file_path


def plot_calibration_interarrival_ecdfs(cleaned_data, output_dir):
    setup_visualization_style()
    output_path = ensure_output_dir(output_dir)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Interarrival Time Empirical CDFs by Period',
                 fontsize=14, fontweight='bold')

    for idx, period_key in enumerate(ALL_PERIOD_KEYS):
        ax = axes[idx // 3, idx % 3]
        df = cleaned_data[period_key]

        if 'interarrival_min' not in df.columns or len(df) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(period_key_to_display_name(period_key))
            continue

        interarrivals = df['interarrival_min'].dropna()
        sorted_data = np.sort(interarrivals)
        ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

        ax.step(sorted_data, ecdf, where='post', linewidth=2)
        ax.fill_between(sorted_data, 0, ecdf, alpha=0.2, step='post')

        mean_ia = interarrivals.mean()
        ax.axvline(mean_ia, color='red', linestyle='--',
                  label=f'Mean = {mean_ia:.2f} min')

        ax.set_xlabel('Interarrival Time (min)')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title(f'{period_key_to_display_name(period_key)}\nn={len(interarrivals)}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)

    plt.tight_layout()
    file_path = output_path / 'calibration_interarrival_ecdfs.png'
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()
    return file_path


def plot_validation_kpi_comparison(validation_results, observed_stats, output_dir):
    setup_visualization_style()
    output_path = ensure_output_dir(output_dir)

    kpi_metrics = [
        ('mean_wait_min', 'Mean Wait Time (min)'),
        ('utilization_per_barista', 'Utilization per Barista'),
        ('throughput', 'Throughput (customers)'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Validation: Simulated vs. Observed KPIs',
                 fontsize=14, fontweight='bold')

    for idx, (kpi_name, kpi_label) in enumerate(kpi_metrics):
        ax = axes[idx]

        periods = list(ALL_PERIOD_KEYS)
        x_pos = np.arange(len(periods))

        sim_means = []
        sim_errors = []
        obs_values = []

        for period_key in periods:
            results = validation_results[period_key]
            ci = results['full_period'][kpi_name]
            sim_means.append(ci['mean'])
            sim_errors.append(ci['upper'] - ci['mean'])

            obs_val = observed_stats.get(period_key, {}).get(kpi_name, np.nan)
            obs_values.append(obs_val if not np.isnan(obs_val) else ci['mean'])

        width = 0.35
        ax.bar(x_pos - width/2, sim_means, width, yerr=sim_errors,
               label='Simulated (mean +/- 95% CI)', alpha=0.8, capsize=5)
        ax.scatter(x_pos + width/2, obs_values, color='red', s=100,
                  marker='D', label='Observed', zorder=3)

        ax.set_xlabel('Period')
        ax.set_ylabel(kpi_label)
        ax.set_title(kpi_label)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([p.replace('_', '\n') for p in periods],
                          rotation=45, ha='right', fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    file_path = output_path / 'validation_kpi_comparison.png'
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()
    return file_path


def plot_staffing_sweep(staffing_results, period_name, output_dir):
    setup_visualization_style()
    output_path = ensure_output_dir(output_dir)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Staffing Sweep Analysis: {period_name}',
                 fontsize=14, fontweight='bold')

    barista_counts = sorted(staffing_results.keys())

    kpi_configs = [
        ('mean_wait_min', 'Mean Wait Time (min)', axes[0, 0]),
        ('utilization_per_barista', 'Utilization per Barista', axes[0, 1]),
        ('p_wait_exceeds_2min', 'P(Wait > 2 min)', axes[1, 0]),
        ('time_avg_queue_length', 'Time-Avg Queue Length', axes[1, 1]),
    ]

    for kpi_name, kpi_label, ax in kpi_configs:
        means = []
        lowers = []
        uppers = []

        for m in barista_counts:
            ci = staffing_results[m]['full_period'][kpi_name]
            means.append(ci['mean'])
            lowers.append(ci['lower'])
            uppers.append(ci['upper'])

        ax.plot(barista_counts, means, 'o-', linewidth=2, markersize=8, label='Mean')
        ax.fill_between(barista_counts, lowers, uppers, alpha=0.3, label='95% CI')

        ax.set_xlabel('Number of Baristas (m)')
        ax.set_ylabel(kpi_label)
        ax.set_title(kpi_label)
        ax.set_xticks(barista_counts)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    file_path = output_path / f'staffing_sweep_{period_name.replace(" ", "_").lower()}.png'
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()
    return file_path


def plot_service_mix_sensitivity(service_mix_results, baseline_p_drink, period_name, output_dir):
    setup_visualization_style()
    output_path = ensure_output_dir(output_dir)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Service Mix Sensitivity: {period_name}',
                 fontsize=14, fontweight='bold')

    deltas = sorted(service_mix_results.keys())
    p_drinks = [min(1.0, max(0.0, baseline_p_drink + d)) for d in deltas]

    kpi_configs = [
        ('mean_wait_min', 'Mean Wait Time (min)', axes[0]),
        ('utilization_per_barista', 'Utilization per Barista', axes[1]),
    ]

    for kpi_name, kpi_label, ax in kpi_configs:
        means = []
        lowers = []
        uppers = []

        for delta in deltas:
            ci = service_mix_results[delta]['full_period'][kpi_name]
            means.append(ci['mean'])
            lowers.append(ci['lower'])
            uppers.append(ci['upper'])

        ax.plot(p_drinks, means, 'o-', linewidth=2, markersize=8, label='Mean')
        ax.fill_between(p_drinks, lowers, uppers, alpha=0.3, label='95% CI')
        ax.axvline(baseline_p_drink, color='red', linestyle='--',
                  label=f'Baseline P(Drink)={baseline_p_drink:.2f}')

        ax.set_xlabel('P(Drink)')
        ax.set_ylabel(kpi_label)
        ax.set_title(kpi_label)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    file_path = output_path / f'service_mix_{period_name.replace(" ", "_").lower()}.png'
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()
    return file_path


def plot_warmup_comparison(warmup_results, period_name, output_dir):
    setup_visualization_style()
    output_path = ensure_output_dir(output_dir)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Warm-up Analysis: {period_name}',
                 fontsize=14, fontweight='bold')

    warmup_durations = sorted(warmup_results.keys())

    full_means = []
    full_errors = []
    post_means = []
    post_errors = []

    for warmup_min in warmup_durations:
        full_ci = warmup_results[warmup_min]['full_period']['mean_wait_min']
        post_ci = warmup_results[warmup_min]['post_warmup']['mean_wait_min']

        full_means.append(full_ci['mean'])
        full_errors.append(full_ci['upper'] - full_ci['mean'])
        post_means.append(post_ci['mean'])
        post_errors.append(post_ci['upper'] - post_ci['mean'])

    x_pos = np.arange(len(warmup_durations))
    width = 0.35

    axes[0].bar(x_pos - width/2, full_means, width, yerr=full_errors,
               label='Full Period', alpha=0.8, capsize=5)
    axes[0].bar(x_pos + width/2, post_means, width, yerr=post_errors,
               label='Post-Warmup', alpha=0.8, capsize=5)
    axes[0].set_xlabel('Warm-up Duration (min)')
    axes[0].set_ylabel('Mean Wait Time (min)')
    axes[0].set_title('Mean Wait Time: Full vs. Post-Warmup')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(warmup_durations)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    differences = [post - full for post, full in zip(post_means, full_means)]
    axes[1].plot(warmup_durations, differences, 'o-', linewidth=2, markersize=8, color='green')
    axes[1].axhline(0, color='red', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Warm-up Duration (min)')
    axes[1].set_ylabel('Difference (Post - Full)')
    axes[1].set_title('Impact of Warm-up Deletion')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    file_path = output_path / f'warmup_analysis_{period_name.replace(" ", "_").lower()}.png'
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()
    return file_path


def plot_threshold_diagnostics(diagnostics, baseline_threshold, output_dir):
    setup_visualization_style()
    output_path = ensure_output_dir(output_dir)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Threshold Diagnostic Analysis: Separation Quality',
                 fontsize=14, fontweight='bold')

    for idx, period_key in enumerate(ALL_PERIOD_KEYS):
        ax = axes[idx // 3, idx % 3]

        if period_key not in diagnostics:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(period_key_to_display_name(period_key))
            continue

        threshold_splits = diagnostics[period_key]['threshold_splits']

        thresholds = [diag['threshold'] for diag in threshold_splits]
        separations = [diag['mean_difference'] for diag in threshold_splits]
        overlaps = [diag['overlap_coefficient'] for diag in threshold_splits]

        ax2 = ax.twinx()

        line1 = ax.plot(thresholds, separations, 'o-', linewidth=2,
                       color='blue', label='Mean Separation')
        line2 = ax2.plot(thresholds, overlaps, 's-', linewidth=2,
                        color='red', label='Overlap Coeff')

        baseline_idx = thresholds.index(baseline_threshold) if baseline_threshold in thresholds else None
        if baseline_idx is not None:
            ax.axvline(baseline_threshold, color='green', linestyle='--',
                      linewidth=2, alpha=0.7)

        ax.set_xlabel('Threshold (min)')
        ax.set_ylabel('Mean Separation (min)', color='blue')
        ax2.set_ylabel('Overlap Coefficient', color='red')
        ax.set_title(period_key_to_display_name(period_key))

        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')

        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='red')

    plt.tight_layout()
    file_path = output_path / 'threshold_diagnostics.png'
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()
    return file_path


def plot_wait_time_ecdf_comparison(validation_results, observed_data, output_dir):
    setup_visualization_style()
    output_path = ensure_output_dir(output_dir)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Wait Time ECDFs: Validation (Observed vs. Simulated)',
                 fontsize=14, fontweight='bold')

    for idx, period_key in enumerate(ALL_PERIOD_KEYS):
        ax = axes[idx // 3, idx % 3]

        if period_key in observed_data:
            df = observed_data[period_key]
            if 'wait_min' in df.columns:
                obs_wait = df['wait_min'].dropna()
                if len(obs_wait) > 0:
                    sorted_obs = np.sort(obs_wait)
                    ecdf_obs = np.arange(1, len(sorted_obs) + 1) / len(sorted_obs)
                    ax.step(sorted_obs, ecdf_obs, where='post', linewidth=2,
                           label='Observed', color='red', alpha=0.7)

        ax.text(0.5, 0.5, 'Simulated data\n(would need replication traces)',
               ha='center', va='center', transform=ax.transAxes,
               fontsize=10, alpha=0.5)

        ax.set_xlabel('Wait Time (min)')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title(period_key_to_display_name(period_key))
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)

    plt.tight_layout()
    file_path = output_path / 'validation_wait_ecdf.png'
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()
    return file_path
