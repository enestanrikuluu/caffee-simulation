import numpy as np
import pandas as pd
from cafe_sim.period_key import period_key_to_display_name


def create_adequacy_test(test_name, passed, metric_name, simulated_value, observed_value, threshold, deviation, message):
    return {
        'test_name': test_name,
        'passed': passed,
        'metric_name': metric_name,
        'simulated_value': simulated_value,
        'observed_value': observed_value,
        'threshold': threshold,
        'deviation': deviation,
        'message': message,
    }


def create_validation_report(period_key, tests, overall_adequate, summary):
    return {
        'period_key': period_key,
        'tests': tests,
        'overall_adequate': overall_adequate,
        'summary': summary,
    }


def check_validation_adequacy(
    period_key,
    simulated_results,
    observed_data,
    mean_wait_tolerance=0.5,
    ecdf_threshold=0.15,
    throughput_tolerance=0.05
):
    tests = []

    if "wait_min" in observed_data.columns:
        obs_wait = observed_data["wait_min"].dropna()
        if len(obs_wait) > 0:
            mean_wait_test = check_mean_agreement(
                "mean_wait_min",
                simulated_results['full_period']["mean_wait_min"],
                float(np.mean(obs_wait)),
                mean_wait_tolerance,
            )
            tests.append(mean_wait_test)

    if "service_time_min" in observed_data.columns:
        obs_throughput = len(observed_data)
        sim_throughput_ci = simulated_results['full_period']["throughput"]
        throughput_test = check_throughput_match(
            sim_throughput_ci,
            obs_throughput,
            throughput_tolerance,
        )
        tests.append(throughput_test)

    if "wait_min" in observed_data.columns:
        obs_wait = observed_data["wait_min"].dropna()
        if len(obs_wait) > 0:
            ecdf_test = check_ecdf_agreement(
                "wait_time_ecdf",
                obs_wait.values,
                simulated_results,
                ecdf_threshold,
            )
            tests.append(ecdf_test)

    overall_adequate = all(test['passed'] for test in tests)

    summary = generate_summary(tests, overall_adequate)

    return create_validation_report(
        period_key=period_key,
        tests=tests,
        overall_adequate=overall_adequate,
        summary=summary,
    )


def check_mean_agreement(metric_name, sim_ci, obs_mean, tolerance):
    sim_mean = sim_ci['mean']
    deviation = abs(sim_mean - obs_mean)

    within_tolerance = deviation <= tolerance
    obs_in_ci = sim_ci['lower'] <= obs_mean <= sim_ci['upper']

    passed = within_tolerance or obs_in_ci

    if passed:
        message = f"Adequate: |{sim_mean:.3f} - {obs_mean:.3f}| = {deviation:.3f} <= {tolerance:.3f}"
    else:
        message = f"Inadequate: |{sim_mean:.3f} - {obs_mean:.3f}| = {deviation:.3f} > {tolerance:.3f}"
        if not obs_in_ci:
            message += f" (obs not in CI [{sim_ci['lower']:.3f}, {sim_ci['upper']:.3f}])"

    return create_adequacy_test(
        test_name="Mean Agreement",
        passed=passed,
        metric_name=metric_name,
        simulated_value=sim_mean,
        observed_value=obs_mean,
        threshold=tolerance,
        deviation=deviation,
        message=message,
    )


def check_throughput_match(sim_throughput_ci, obs_throughput, tolerance_pct):
    sim_mean = sim_throughput_ci['mean']
    deviation_pct = abs(sim_mean - obs_throughput) / obs_throughput if obs_throughput > 0 else 0

    passed = deviation_pct <= tolerance_pct

    if passed:
        message = f"Adequate: |{sim_mean:.1f} - {obs_throughput}| / {obs_throughput} = {deviation_pct:.3f} <= {tolerance_pct:.3f}"
    else:
        message = f"Inadequate: |{sim_mean:.1f} - {obs_throughput}| / {obs_throughput} = {deviation_pct:.3f} > {tolerance_pct:.3f}"

    return create_adequacy_test(
        test_name="Throughput Match",
        passed=passed,
        metric_name="throughput",
        simulated_value=sim_mean,
        observed_value=float(obs_throughput),
        threshold=tolerance_pct,
        deviation=deviation_pct,
        message=message,
    )


def check_ecdf_agreement(metric_name, obs_values, sim_results, ecdf_threshold):
    sim_mean = sim_results['full_period']["mean_wait_min"]['mean']
    sim_std = sim_results['full_period']["mean_wait_min"]['std']

    synthetic_sim_samples = np.random.normal(sim_mean, sim_std, size=len(obs_values))
    synthetic_sim_samples = np.maximum(0, synthetic_sim_samples)

    max_deviation = compute_ecdf_max_deviation(obs_values, synthetic_sim_samples)

    passed = max_deviation <= ecdf_threshold

    if passed:
        message = f"Adequate: max ECDF deviation = {max_deviation:.3f} <= {ecdf_threshold:.3f}"
    else:
        message = f"Inadequate: max ECDF deviation = {max_deviation:.3f} > {ecdf_threshold:.3f}"

    return create_adequacy_test(
        test_name="ECDF Agreement",
        passed=passed,
        metric_name=metric_name,
        simulated_value=sim_mean,
        observed_value=float(np.mean(obs_values)),
        threshold=ecdf_threshold,
        deviation=max_deviation,
        message=message,
    )


def compute_ecdf_max_deviation(obs_values, sim_values):
    all_values = np.concatenate([obs_values, sim_values])
    eval_points = np.sort(np.unique(all_values))

    obs_ecdf = np.array([np.mean(obs_values <= x) for x in eval_points])
    sim_ecdf = np.array([np.mean(sim_values <= x) for x in eval_points])

    max_deviation = np.max(np.abs(obs_ecdf - sim_ecdf))
    return float(max_deviation)


def generate_summary(tests, overall_adequate):
    passed_count = sum(1 for t in tests if t['passed'])
    total_count = len(tests)

    summary = f"Validation Adequacy: {'PASS' if overall_adequate else 'FAIL'} ({passed_count}/{total_count} tests passed)\n"
    summary += "\n"

    for test in tests:
        status = "+" if test['passed'] else "x"
        summary += f"{status} {test['test_name']} ({test['metric_name']}): {test['message']}\n"

    return summary


def generate_full_report(reports, mean_wait_tolerance=0.5, ecdf_threshold=0.15, throughput_tolerance=0.05):
    report_text = "="*80 + "\n"
    report_text += "VALIDATION ADEQUACY REPORT\n"
    report_text += "="*80 + "\n\n"

    overall_pass = all(r['overall_adequate'] for r in reports.values())
    report_text += f"Overall Assessment: {'ADEQUATE' if overall_pass else 'INADEQUATE'}\n"
    report_text += f"Periods Passed: {sum(1 for r in reports.values() if r['overall_adequate'])}/{len(reports)}\n\n"

    for period_key, report in reports.items():
        report_text += "-"*80 + "\n"
        report_text += f"{period_key_to_display_name(period_key)}\n"
        report_text += "-"*80 + "\n"
        report_text += report['summary'] + "\n"

    report_text += "="*80 + "\n"
    report_text += "ADEQUACY THRESHOLDS\n"
    report_text += "="*80 + "\n"
    report_text += f"Mean Wait Tolerance: +/-{mean_wait_tolerance} min\n"
    report_text += f"ECDF Max Deviation: <={ecdf_threshold}\n"
    report_text += f"Throughput Tolerance: +/-{throughput_tolerance*100:.1f}%\n"

    return report_text
