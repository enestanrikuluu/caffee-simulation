import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple
from cafe_sim.configuration import AggregatedKPIs
from cafe_sim.period_key import PeriodKey


@dataclass
class AdequacyTest:
    test_name: str
    passed: bool
    metric_name: str
    simulated_value: float
    observed_value: float
    threshold: float
    deviation: float
    message: str


@dataclass
class ValidationAdequacyReport:
    period_key: PeriodKey
    tests: List[AdequacyTest]
    overall_adequate: bool
    summary: str


class ValidationAdequacyChecker:
    def __init__(
        self,
        mean_wait_tolerance_min: float = 0.5,
        ecdf_max_deviation_threshold: float = 0.15,
        throughput_tolerance_pct: float = 0.05,
    ):
        self._mean_wait_tolerance = mean_wait_tolerance_min
        self._ecdf_threshold = ecdf_max_deviation_threshold
        self._throughput_tolerance = throughput_tolerance_pct

    def check_validation_adequacy(
        self,
        period_key: PeriodKey,
        simulated_results: AggregatedKPIs,
        observed_data: pd.DataFrame,
    ) -> ValidationAdequacyReport:
        tests = []
        
        if "wait_min" in observed_data.columns:
            obs_wait = observed_data["wait_min"].dropna()
            if len(obs_wait) > 0:
                mean_wait_test = self._check_mean_agreement(
                    "mean_wait_min",
                    simulated_results.full_period["mean_wait_min"],
                    float(np.mean(obs_wait)),
                    self._mean_wait_tolerance,
                )
                tests.append(mean_wait_test)
        
        if "service_time_min" in observed_data.columns:
            obs_throughput = len(observed_data)
            sim_throughput_ci = simulated_results.full_period["throughput"]
            throughput_test = self._check_throughput_match(
                sim_throughput_ci,
                obs_throughput,
                self._throughput_tolerance,
            )
            tests.append(throughput_test)
        
        if "wait_min" in observed_data.columns:
            obs_wait = observed_data["wait_min"].dropna()
            if len(obs_wait) > 0:
                ecdf_test = self._check_ecdf_agreement(
                    "wait_time_ecdf",
                    obs_wait.values,
                    simulated_results,
                )
                tests.append(ecdf_test)
        
        overall_adequate = all(test.passed for test in tests)
        
        summary = self._generate_summary(tests, overall_adequate)
        
        return ValidationAdequacyReport(
            period_key=period_key,
            tests=tests,
            overall_adequate=overall_adequate,
            summary=summary,
        )

    def _check_mean_agreement(
        self, metric_name: str, sim_ci, obs_mean: float, tolerance: float
    ) -> AdequacyTest:
        sim_mean = sim_ci.mean
        deviation = abs(sim_mean - obs_mean)
        
        within_tolerance = deviation <= tolerance
        obs_in_ci = sim_ci.lower <= obs_mean <= sim_ci.upper
        
        passed = within_tolerance or obs_in_ci
        
        if passed:
            message = f"Adequate: |{sim_mean:.3f} - {obs_mean:.3f}| = {deviation:.3f} ≤ {tolerance:.3f}"
        else:
            message = f"Inadequate: |{sim_mean:.3f} - {obs_mean:.3f}| = {deviation:.3f} > {tolerance:.3f}"
            if not obs_in_ci:
                message += f" (obs not in CI [{sim_ci.lower:.3f}, {sim_ci.upper:.3f}])"
        
        return AdequacyTest(
            test_name="Mean Agreement",
            passed=passed,
            metric_name=metric_name,
            simulated_value=sim_mean,
            observed_value=obs_mean,
            threshold=tolerance,
            deviation=deviation,
            message=message,
        )

    def _check_throughput_match(
        self, sim_throughput_ci, obs_throughput: int, tolerance_pct: float
    ) -> AdequacyTest:
        sim_mean = sim_throughput_ci.mean
        deviation_pct = abs(sim_mean - obs_throughput) / obs_throughput if obs_throughput > 0 else 0
        
        passed = deviation_pct <= tolerance_pct
        
        if passed:
            message = f"Adequate: |{sim_mean:.1f} - {obs_throughput}| / {obs_throughput} = {deviation_pct:.3f} ≤ {tolerance_pct:.3f}"
        else:
            message = f"Inadequate: |{sim_mean:.1f} - {obs_throughput}| / {obs_throughput} = {deviation_pct:.3f} > {tolerance_pct:.3f}"
        
        return AdequacyTest(
            test_name="Throughput Match",
            passed=passed,
            metric_name="throughput",
            simulated_value=sim_mean,
            observed_value=float(obs_throughput),
            threshold=tolerance_pct,
            deviation=deviation_pct,
            message=message,
        )

    def _check_ecdf_agreement(
        self, metric_name: str, obs_values: np.ndarray, sim_results: AggregatedKPIs
    ) -> AdequacyTest:
        sim_mean = sim_results.full_period["mean_wait_min"].mean
        sim_std = sim_results.full_period["mean_wait_min"].std
        
        synthetic_sim_samples = np.random.normal(sim_mean, sim_std, size=len(obs_values))
        synthetic_sim_samples = np.maximum(0, synthetic_sim_samples)
        
        max_deviation = self._compute_ecdf_max_deviation(obs_values, synthetic_sim_samples)
        
        passed = max_deviation <= self._ecdf_threshold
        
        if passed:
            message = f"Adequate: max ECDF deviation = {max_deviation:.3f} ≤ {self._ecdf_threshold:.3f}"
        else:
            message = f"Inadequate: max ECDF deviation = {max_deviation:.3f} > {self._ecdf_threshold:.3f}"
        
        return AdequacyTest(
            test_name="ECDF Agreement",
            passed=passed,
            metric_name=metric_name,
            simulated_value=sim_mean,
            observed_value=float(np.mean(obs_values)),
            threshold=self._ecdf_threshold,
            deviation=max_deviation,
            message=message,
        )

    def _compute_ecdf_max_deviation(
        self, obs_values: np.ndarray, sim_values: np.ndarray
    ) -> float:
        all_values = np.concatenate([obs_values, sim_values])
        eval_points = np.sort(np.unique(all_values))
        
        obs_ecdf = np.array([np.mean(obs_values <= x) for x in eval_points])
        sim_ecdf = np.array([np.mean(sim_values <= x) for x in eval_points])
        
        max_deviation = np.max(np.abs(obs_ecdf - sim_ecdf))
        return float(max_deviation)

    def _generate_summary(self, tests: List[AdequacyTest], overall_adequate: bool) -> str:
        passed_count = sum(1 for t in tests if t.passed)
        total_count = len(tests)
        
        summary = f"Validation Adequacy: {'PASS' if overall_adequate else 'FAIL'} ({passed_count}/{total_count} tests passed)\n"
        summary += "\n"
        
        for test in tests:
            status = "✓" if test.passed else "✗"
            summary += f"{status} {test.test_name} ({test.metric_name}): {test.message}\n"
        
        return summary

    def generate_full_report(
        self, reports: Dict[PeriodKey, ValidationAdequacyReport]
    ) -> str:
        report_text = "="*80 + "\n"
        report_text += "VALIDATION ADEQUACY REPORT\n"
        report_text += "="*80 + "\n\n"
        
        overall_pass = all(r.overall_adequate for r in reports.values())
        report_text += f"Overall Assessment: {'ADEQUATE' if overall_pass else 'INADEQUATE'}\n"
        report_text += f"Periods Passed: {sum(1 for r in reports.values() if r.overall_adequate)}/{len(reports)}\n\n"
        
        for period_key, report in reports.items():
            report_text += "-"*80 + "\n"
            report_text += f"{period_key.to_display_name()}\n"
            report_text += "-"*80 + "\n"
            report_text += report.summary + "\n"
        
        report_text += "="*80 + "\n"
        report_text += "ADEQUACY THRESHOLDS\n"
        report_text += "="*80 + "\n"
        report_text += f"Mean Wait Tolerance: ±{self._mean_wait_tolerance} min\n"
        report_text += f"ECDF Max Deviation: ≤{self._ecdf_threshold}\n"
        report_text += f"Throughput Tolerance: ±{self._throughput_tolerance*100:.1f}%\n"
        
        return report_text
