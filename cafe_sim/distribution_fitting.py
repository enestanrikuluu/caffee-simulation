import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable
from enum import Enum
import warnings


class TheoreticalDistribution(Enum):
    EXPONENTIAL = "exponential"
    GAMMA = "gamma"
    WEIBULL = "weibull"
    LOGNORMAL = "lognormal"
    NORMAL = "normal"
    UNIFORM = "uniform"


@dataclass
class FitResult:
    distribution: TheoreticalDistribution
    parameters: Dict[str, float]
    log_likelihood: float
    aic: float
    bic: float
    chi_square_statistic: float
    chi_square_p_value: float
    ks_statistic: float
    ks_p_value: float
    ad_statistic: float
    ad_critical_values: Dict[str, float]
    sample_size: int
    rank: int


@dataclass
class DistributionFittingReport:
    variable_name: str
    sample_size: int
    sample_mean: float
    sample_std: float
    sample_min: float
    sample_max: float
    fit_results: List[FitResult]
    recommended_distribution: TheoreticalDistribution
    recommendation_reason: str


class DistributionFitter:
    def __init__(self, num_bins_chi_square: int = 10, min_expected_frequency: float = 5.0):
        self._num_bins = num_bins_chi_square
        self._min_expected = min_expected_frequency
    
    def fit_all_distributions(
        self, data: np.ndarray, variable_name: str = "unknown"
    ) -> DistributionFittingReport:
        data = data[~np.isnan(data)]
        data = data[data >= 0]
        
        if len(data) < 30:
            raise ValueError(f"Insufficient data: {len(data)} samples (need at least 30)")
        
        sample_stats = {
            "size": len(data),
            "mean": float(np.mean(data)),
            "std": float(np.std(data, ddof=1)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
        }
        
        fit_results = []
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            if sample_stats["min"] >= 0:
                fit_results.append(self._fit_exponential(data))
                fit_results.append(self._fit_gamma(data))
                fit_results.append(self._fit_weibull(data))
                
                if sample_stats["min"] > 0:
                    fit_results.append(self._fit_lognormal(data))
            
            fit_results.append(self._fit_normal(data))
            fit_results.append(self._fit_uniform(data))
        
        fit_results.sort(key=lambda x: x.aic)
        for rank, result in enumerate(fit_results, start=1):
            result.rank = rank
        
        recommended = self._select_recommended_distribution(fit_results, data)
        
        return DistributionFittingReport(
            variable_name=variable_name,
            sample_size=sample_stats["size"],
            sample_mean=sample_stats["mean"],
            sample_std=sample_stats["std"],
            sample_min=sample_stats["min"],
            sample_max=sample_stats["max"],
            fit_results=fit_results,
            recommended_distribution=recommended.distribution,
            recommendation_reason=self._generate_recommendation_reason(recommended, fit_results),
        )
    
    def _fit_exponential(self, data: np.ndarray) -> FitResult:
        rate = 1.0 / np.mean(data)
        params = {"rate": rate}
        
        log_lik = np.sum(stats.expon.logpdf(data, scale=1/rate))
        k = 1
        n = len(data)
        aic = 2 * k - 2 * log_lik
        bic = k * np.log(n) - 2 * log_lik
        
        chi2_stat, chi2_p = self._chi_square_test(
            data, lambda x: stats.expon.cdf(x, scale=1/rate)
        )
        
        ks_stat, ks_p = stats.kstest(data, lambda x: stats.expon.cdf(x, scale=1/rate))
        
        ad_result = stats.anderson(data, dist="expon")
        ad_stat = ad_result.statistic
        ad_crit = {
            "15%": ad_result.critical_values[0],
            "10%": ad_result.critical_values[1],
            "5%": ad_result.critical_values[2],
            "2.5%": ad_result.critical_values[3],
            "1%": ad_result.critical_values[4],
        }
        
        return FitResult(
            distribution=TheoreticalDistribution.EXPONENTIAL,
            parameters=params,
            log_likelihood=log_lik,
            aic=aic,
            bic=bic,
            chi_square_statistic=chi2_stat,
            chi_square_p_value=chi2_p,
            ks_statistic=ks_stat,
            ks_p_value=ks_p,
            ad_statistic=ad_stat,
            ad_critical_values=ad_crit,
            sample_size=n,
            rank=0,
        )
    
    def _fit_gamma(self, data: np.ndarray) -> FitResult:
        try:
            shape, loc, scale = stats.gamma.fit(data, floc=0)
        except (ValueError, RuntimeError):
            mean_val = np.mean(data)
            var_val = np.var(data, ddof=1)
            if var_val < 1e-10 or mean_val < 1e-10:
                var_val = mean_val * 0.1
            shape = max(0.1, (mean_val ** 2) / var_val)
            scale = max(0.01, var_val / mean_val)
        
        params = {"shape": shape, "scale": scale}
        
        log_lik = np.sum(stats.gamma.logpdf(data, shape, loc=0, scale=scale))
        k = 2
        n = len(data)
        aic = 2 * k - 2 * log_lik
        bic = k * np.log(n) - 2 * log_lik
        
        chi2_stat, chi2_p = self._chi_square_test(
            data, lambda x: stats.gamma.cdf(x, shape, loc=0, scale=scale)
        )
        
        ks_stat, ks_p = stats.kstest(
            data, lambda x: stats.gamma.cdf(x, shape, loc=0, scale=scale)
        )
        
        ad_stat, ad_crit = self._anderson_darling_generic(
            data, lambda x: stats.gamma.cdf(x, shape, loc=0, scale=scale)
        )
        
        return FitResult(
            distribution=TheoreticalDistribution.GAMMA,
            parameters=params,
            log_likelihood=log_lik,
            aic=aic,
            bic=bic,
            chi_square_statistic=chi2_stat,
            chi_square_p_value=chi2_p,
            ks_statistic=ks_stat,
            ks_p_value=ks_p,
            ad_statistic=ad_stat,
            ad_critical_values=ad_crit,
            sample_size=n,
            rank=0,
        )
    
    def _fit_weibull(self, data: np.ndarray) -> FitResult:
        try:
            shape, loc, scale = stats.weibull_min.fit(data, floc=0)
        except (ValueError, RuntimeError):
            mean_val = np.mean(data)
            shape = 2.0
            scale = mean_val / np.sqrt(np.pi / 2)
        
        params = {"shape": shape, "scale": scale}
        
        log_lik = np.sum(stats.weibull_min.logpdf(data, shape, loc=0, scale=scale))
        k = 2
        n = len(data)
        aic = 2 * k - 2 * log_lik
        bic = k * np.log(n) - 2 * log_lik
        
        chi2_stat, chi2_p = self._chi_square_test(
            data, lambda x: stats.weibull_min.cdf(x, shape, loc=0, scale=scale)
        )
        
        ks_stat, ks_p = stats.kstest(
            data, lambda x: stats.weibull_min.cdf(x, shape, loc=0, scale=scale)
        )
        
        ad_stat, ad_crit = self._anderson_darling_generic(
            data, lambda x: stats.weibull_min.cdf(x, shape, loc=0, scale=scale)
        )
        
        return FitResult(
            distribution=TheoreticalDistribution.WEIBULL,
            parameters=params,
            log_likelihood=log_lik,
            aic=aic,
            bic=bic,
            chi_square_statistic=chi2_stat,
            chi_square_p_value=chi2_p,
            ks_statistic=ks_stat,
            ks_p_value=ks_p,
            ad_statistic=ad_stat,
            ad_critical_values=ad_crit,
            sample_size=n,
            rank=0,
        )
    
    def _fit_lognormal(self, data: np.ndarray) -> FitResult:
        try:
            shape, loc, scale = stats.lognorm.fit(data, floc=0)
        except (ValueError, RuntimeError):
            log_data = np.log(data[data > 0])
            shape = np.std(log_data, ddof=1)
            scale = np.exp(np.mean(log_data))
        
        params = {"shape": shape, "scale": scale}
        
        log_lik = np.sum(stats.lognorm.logpdf(data, shape, loc=0, scale=scale))
        k = 2
        n = len(data)
        aic = 2 * k - 2 * log_lik
        bic = k * np.log(n) - 2 * log_lik
        
        chi2_stat, chi2_p = self._chi_square_test(
            data, lambda x: stats.lognorm.cdf(x, shape, loc=0, scale=scale)
        )
        
        ks_stat, ks_p = stats.kstest(
            data, lambda x: stats.lognorm.cdf(x, shape, loc=0, scale=scale)
        )
        
        ad_stat, ad_crit = self._anderson_darling_generic(
            data, lambda x: stats.lognorm.cdf(x, shape, loc=0, scale=scale)
        )
        
        return FitResult(
            distribution=TheoreticalDistribution.LOGNORMAL,
            parameters=params,
            log_likelihood=log_lik,
            aic=aic,
            bic=bic,
            chi_square_statistic=chi2_stat,
            chi_square_p_value=chi2_p,
            ks_statistic=ks_stat,
            ks_p_value=ks_p,
            ad_statistic=ad_stat,
            ad_critical_values=ad_crit,
            sample_size=n,
            rank=0,
        )
    
    def _fit_normal(self, data: np.ndarray) -> FitResult:
        mu, sigma = np.mean(data), np.std(data, ddof=1)
        params = {"mu": mu, "sigma": sigma}
        
        log_lik = np.sum(stats.norm.logpdf(data, loc=mu, scale=sigma))
        k = 2
        n = len(data)
        aic = 2 * k - 2 * log_lik
        bic = k * np.log(n) - 2 * log_lik
        
        chi2_stat, chi2_p = self._chi_square_test(
            data, lambda x: stats.norm.cdf(x, loc=mu, scale=sigma)
        )
        
        ks_stat, ks_p = stats.kstest(data, lambda x: stats.norm.cdf(x, loc=mu, scale=sigma))
        
        ad_result = stats.anderson(data, dist="norm")
        ad_stat = ad_result.statistic
        ad_crit = {
            "15%": ad_result.critical_values[0],
            "10%": ad_result.critical_values[1],
            "5%": ad_result.critical_values[2],
            "2.5%": ad_result.critical_values[3],
            "1%": ad_result.critical_values[4],
        }
        
        return FitResult(
            distribution=TheoreticalDistribution.NORMAL,
            parameters=params,
            log_likelihood=log_lik,
            aic=aic,
            bic=bic,
            chi_square_statistic=chi2_stat,
            chi_square_p_value=chi2_p,
            ks_statistic=ks_stat,
            ks_p_value=ks_p,
            ad_statistic=ad_stat,
            ad_critical_values=ad_crit,
            sample_size=n,
            rank=0,
        )
    
    def _fit_uniform(self, data: np.ndarray) -> FitResult:
        a, b = np.min(data), np.max(data)
        params = {"a": a, "b": b}
        
        log_lik = np.sum(stats.uniform.logpdf(data, loc=a, scale=b-a))
        k = 2
        n = len(data)
        aic = 2 * k - 2 * log_lik
        bic = k * np.log(n) - 2 * log_lik
        
        chi2_stat, chi2_p = self._chi_square_test(
            data, lambda x: stats.uniform.cdf(x, loc=a, scale=b-a)
        )
        
        ks_stat, ks_p = stats.kstest(data, lambda x: stats.uniform.cdf(x, loc=a, scale=b-a))
        
        ad_stat, ad_crit = self._anderson_darling_generic(
            data, lambda x: stats.uniform.cdf(x, loc=a, scale=b-a)
        )
        
        return FitResult(
            distribution=TheoreticalDistribution.UNIFORM,
            parameters=params,
            log_likelihood=log_lik,
            aic=aic,
            bic=bic,
            chi_square_statistic=chi2_stat,
            chi_square_p_value=chi2_p,
            ks_statistic=ks_stat,
            ks_p_value=ks_p,
            ad_statistic=ad_stat,
            ad_critical_values=ad_crit,
            sample_size=n,
            rank=0,
        )
    
    def _chi_square_test(
        self, data: np.ndarray, cdf_func: Callable[[np.ndarray], np.ndarray]
    ) -> Tuple[float, float]:
        n = len(data)
        num_bins = min(self._num_bins, int(np.sqrt(n)))
        
        bin_edges = np.linspace(np.min(data), np.max(data), num_bins + 1)
        observed_freq, _ = np.histogram(data, bins=bin_edges)
        
        expected_probs = np.diff(cdf_func(bin_edges))
        expected_freq = n * expected_probs
        
        mask = expected_freq >= self._min_expected
        if np.sum(mask) < 3:
            return np.nan, np.nan
        
        observed_freq = observed_freq[mask]
        expected_freq = expected_freq[mask]
        
        chi2_stat = np.sum((observed_freq - expected_freq) ** 2 / expected_freq)
        
        df = len(observed_freq) - 1 - 2
        if df <= 0:
            return chi2_stat, np.nan
        
        p_value = 1 - stats.chi2.cdf(chi2_stat, df)
        
        return float(chi2_stat), float(p_value)
    
    def _anderson_darling_generic(
        self, data: np.ndarray, cdf_func: Callable[[np.ndarray], np.ndarray]
    ) -> Tuple[float, Dict[str, float]]:
        sorted_data = np.sort(data)
        n = len(sorted_data)
        
        cdf_values = cdf_func(sorted_data)
        cdf_values = np.clip(cdf_values, 1e-10, 1 - 1e-10)
        
        i = np.arange(1, n + 1)
        ad_stat = -n - np.sum((2*i - 1) / n * (np.log(cdf_values) + np.log(1 - cdf_values[::-1])))
        
        critical_values = {
            "15%": 1.610,
            "10%": 1.933,
            "5%": 2.492,
            "2.5%": 3.070,
            "1%": 3.857,
        }
        
        return float(ad_stat), critical_values
    
    def _select_recommended_distribution(
        self, fit_results: List[FitResult], data: np.ndarray
    ) -> FitResult:
        ks_alpha = 0.05
        acceptable_fits = [
            result for result in fit_results
            if not np.isnan(result.ks_p_value) and result.ks_p_value > ks_alpha
        ]
        
        if acceptable_fits:
            return acceptable_fits[0]
        
        return fit_results[0]
    
    def _generate_recommendation_reason(
        self, recommended: FitResult, all_results: List[FitResult]
    ) -> str:
        reason = f"Rank #{recommended.rank} by AIC ({recommended.aic:.2f})"
        
        if not np.isnan(recommended.ks_p_value):
            if recommended.ks_p_value > 0.05:
                reason += f", K-S test pass (p={recommended.ks_p_value:.3f})"
            else:
                reason += f", K-S test marginal (p={recommended.ks_p_value:.3f})"
        
        if recommended.ad_statistic < recommended.ad_critical_values["5%"]:
            reason += ", A-D test pass"
        
        return reason
    
    def generate_text_report(self, reports: Dict[str, DistributionFittingReport]) -> str:
        text = "=" * 100 + "\n"
        text += "DISTRIBUTION FITTING REPORT\n"
        text += "=" * 100 + "\n\n"
        
        for var_name, report in reports.items():
            text += "-" * 100 + "\n"
            text += f"VARIABLE: {report.variable_name}\n"
            text += "-" * 100 + "\n"
            text += f"Sample Size: {report.sample_size}\n"
            text += f"Sample Mean: {report.sample_mean:.4f}\n"
            text += f"Sample Std:  {report.sample_std:.4f}\n"
            text += f"Sample Range: [{report.sample_min:.4f}, {report.sample_max:.4f}]\n"
            text += f"\nRECOMMENDED: {report.recommended_distribution.value.upper()}\n"
            text += f"Reason: {report.recommendation_reason}\n\n"
            
            text += f"{'Rank':<6} {'Distribution':<15} {'AIC':<10} {'BIC':<10} "
            text += f"{'K-S p-val':<12} {'Chi² p-val':<12} {'A-D stat':<10}\n"
            text += "-" * 100 + "\n"
            
            for result in report.fit_results:
                ks_p_str = f"{result.ks_p_value:.4f}" if not np.isnan(result.ks_p_value) else "N/A"
                chi2_p_str = f"{result.chi_square_p_value:.4f}" if not np.isnan(result.chi_square_p_value) else "N/A"
                
                text += f"{result.rank:<6} {result.distribution.value:<15} "
                text += f"{result.aic:<10.2f} {result.bic:<10.2f} "
                text += f"{ks_p_str:<12} {chi2_p_str:<12} {result.ad_statistic:<10.3f}\n"
                
                param_str = ", ".join([f"{k}={v:.4f}" for k, v in result.parameters.items()])
                text += f"       Parameters: {param_str}\n"
            
            text += "\n"
        
        text += "=" * 100 + "\n"
        text += "INTERPRETATION GUIDE\n"
        text += "=" * 100 + "\n"
        text += "AIC/BIC: Lower is better (penalizes model complexity)\n"
        text += "K-S p-value: >0.05 indicates good fit (fail to reject H₀)\n"
        text += "Chi² p-value: >0.05 indicates good fit (fail to reject H₀)\n"
        text += "Anderson-Darling: Statistic < critical value indicates good fit\n"
        text += "Rank: Based on AIC (1 = best fit)\n"
        
        return text
