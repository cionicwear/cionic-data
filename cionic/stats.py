from typing import Dict, Tuple

import numpy as np
from scipy.special import gamma

MAD_SCALING_FACTOR = 1.4826


def compute_mae(true: np.ndarray, pred: np.ndarray) -> float:
    """Compute Mean Absolute Error between true and predicted values.

    Args:
        true (np.ndarray): Array of true/reference values.
        pred (np.ndarray): Array of predicted values.

    Returns:
        float: Mean absolute error.
    """
    return np.mean(np.abs(true - pred))


def compute_mape(true: np.ndarray, pred: np.ndarray) -> float:
    """Compute Mean Absolute Percentage Error between true and predicted values.

    Args:
        true (np.ndarray): Array of true/reference values.
        pred (np.ndarray): Array of predicted values.

    Returns:
        float: Mean absolute percentage error as a percentage (0-100).
    """
    return np.mean(np.abs((true - pred) / true)) * 100


def compute_rmse(true: np.ndarray, pred: np.ndarray) -> float:
    """Compute Root Mean Square Error between true and predicted values.

    Args:
        true (np.ndarray): Array of true/reference values.
        pred (np.ndarray): Array of predicted values.

    Returns:
        float: Root mean square error.
    """
    return np.sqrt(np.mean((true - pred) ** 2))


def compute_absolute_error(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute element-wise absolute errors between true and predicted values.

    Args:
        true (np.ndarray): Array of true/reference values.
        pred (np.ndarray): Array of predicted values.

    Returns:
        np.ndarray: Array of absolute errors for each prediction.
    """
    return np.abs(true - pred)


def compute_absolute_percentage_error(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute element-wise absolute percentage errors between true and predicted vals.

    Args:
        true (np.ndarray): Array of true/reference values.
        pred (np.ndarray): Array of predicted values.

    Returns:
        np.ndarray: Array of absolute percentage errors (0-100) for each prediction.
    """
    return np.abs((true - pred) / true) * 100


def compute_percentage_within_threshold(
    true: np.ndarray, pred: np.ndarray, threshold_percent: float
) -> float:
    """Compute percentage of predictions within a specified error threshold.

    Args:
        true (np.ndarray): Array of true/reference values.
        pred (np.ndarray): Array of predicted values.
        threshold_percent: Error threshold as a percentage (e.g., 5.0 for 5%).

    Returns:
        float: Percentage of predictions within the threshold (0-100).
    """
    abs_percent_errors = np.abs(true - pred) / true * 100
    within_threshold = abs_percent_errors <= threshold_percent
    return np.mean(within_threshold) * 100


def compute_percentage_within_five_percent(true: np.ndarray, pred: np.ndarray) -> float:
    """Compute percentage of predictions within 5% error threshold.

    Args:
        true (np.ndarray): Array of true/reference values.
        pred (np.ndarray): Array of predicted values.

    Returns:
        float: Percentage of predictions within 5% error (0-100).
    """
    return compute_percentage_within_threshold(true, pred, 5)


def compute_percentage_with_ten_percent(true: np.ndarray, pred: np.ndarray) -> float:
    """Compute percentage of predictions within 10% error threshold.

    Args:
        true (np.ndarray): Array of true/reference values.
        pred (np.ndarray): Array of predicted values.

    Returns:
        float: Percentage of predictions within 10% error (0-100).
    """
    return compute_percentage_within_threshold(true, pred, 10)


def c4_correction(n: int) -> float:
    """
    Compute the small-sample bias correction factor for the standard deviation.

    This correction factor, c4(n), is used to obtain an unbiased estimate of the
    sample standard deviation:
        unbiased_SD = sample_SD / c4(n)

    Args:
        n (int): Sample size (must be >= 2).

    Returns:
        float: The c4 correction factor. Returns np.nan if n < 2.
    """
    if n < 2:
        return np.nan
    return np.sqrt(2 / (n - 1)) * (gamma(n / 2) / gamma((n - 1) / 2))


def robust_sd(x: np.ndarray) -> float:
    """
    Compute a robust estimate of the standard deviation using the
    median absolute deviation (MAD). The MAD is scaled by a constant
    factor to make it comparable to the standard deviation.

    This method is less sensitive to outliers than the standard deviation.

    Args:
        x (np.ndarray): 1D array of values.

    Returns:
        float: Robust standard deviation estimate.
    """
    mad = np.median(np.abs(x - np.median(x)))
    return MAD_SCALING_FACTOR * mad


def coefficient_of_variation(sd: float, mean: float) -> float:
    """
    Calculate the coefficient of variation (CV) as a percentage.

    Args:
        sd (float): Standard deviation.
        mean (float): Mean value.

    Returns:
        float: Coefficient of variation in percent. Returns np.nan if mean is zero.
    """
    return (sd / mean) * 100 if mean != 0 else np.nan


def bootstrap_ci(
    func, x: np.ndarray, n_boot: int = 2000, ci: float = 95
) -> Tuple[Tuple[float, float], float]:
    """
    Compute a bootstrap confidence interval for a statistic.

    Args:
        func (callable): Function that takes a 1D array and returns a scalar statistic.
        x (np.ndarray): 1D array of data to sample from.
        n_boot (int, optional): Number of bootstrap samples. Default is 2000.
        ci (float, optional): Confidence interval width (e.g., 95 for 95% CI).

    Returns:
        Tuple[Tuple[float, float], float]: A tuple where the first element is a tuple
        containing the (lower, upper) confidence interval bounds, and the second element
        is the statistic computed on the original data.
    """
    rng = np.random.default_rng()
    boot_stats = []
    n = len(x)
    for _ in range(n_boot):
        sample = rng.choice(x, size=n, replace=True)
        boot_stats.append(func(sample))
    lower = np.percentile(boot_stats, (100 - ci) / 2)
    upper = np.percentile(boot_stats, 100 - (100 - ci) / 2)
    return (lower, upper), func(x)


def summarize_metric_distribution(
    values: np.ndarray, drop_worst: bool = True, n_boot: int = 2000
) -> Dict:
    """
    Summarize the distribution of a 1D array of metric values (e.g., stride times, ROM).

    Computes mean, standard deviation (SD), unbiased SD (c4 corrected), robust SD (MAD),
    coefficient of variation (CV), robust CV, bootstrap confidence intervals for SD
    and CV, and sensitivity analysis by dropping the worst outlier.

    Args:
        values (np.ndarray): 1D array of metric values.
        drop_worst (bool, optional): If True, also compute summary with worst outlier
            removed. Default is True.
        n_boot (int, optional): Number of bootstrap samples for confidence intervals.
            Default is 2000.

    Returns:
        Dict: Dictionary containing summary statistics:
            - n: Number of values
            - mean: Mean of values
            - sd: Sample standard deviation
            - sd_unbiased: Unbiased SD (c4 corrected)
            - robust_sd: Robust SD (MAD)
            - cv: Coefficient of variation (unbiased SD / mean)
            - robust_cv: Robust coefficient of variation (robust SD / mean)
            - sd_ci: Bootstrap confidence interval for SD
            - cv_ci: Bootstrap confidence interval for CV
            - drop_worst: (if drop_worst=True and n>2) dict with summary after removing
                worst outlier
    """
    values = np.asarray(values).astype(float)
    values = values[~np.isnan(values)]
    n = len(values)
    if n < 2:
        raise ValueError("Need at least 2 values for variability metrics.")

    mean_val = np.mean(values)
    sd_val = np.std(values, ddof=1)  # sample SD
    c4 = c4_correction(n)
    sd_unbiased = sd_val / c4 if not np.isnan(c4) else np.nan
    robust_sd_val = robust_sd(values)

    cv_val = coefficient_of_variation(sd_unbiased, mean_val)
    robust_cv_val = coefficient_of_variation(robust_sd_val, mean_val)

    # Bootstrap CIs
    sd_ci, _ = bootstrap_ci(lambda arr: np.std(arr, ddof=1), values, n_boot=n_boot)
    cv_ci, _ = bootstrap_ci(
        lambda arr: coefficient_of_variation(np.std(arr, ddof=1), np.mean(arr)),
        values,
        n_boot=n_boot,
    )

    result = {
        "n": n,
        "mean": mean_val,
        "sd": sd_val,
        "sd_unbiased": sd_unbiased,
        "robust_sd": robust_sd_val,
        "cv": cv_val,
        "robust_cv": robust_cv_val,
        "sd_ci": sd_ci,
        "cv_ci": cv_ci,
    }

    # Sensitivity: drop worst outlier by absolute deviation
    if drop_worst and n > 2:
        abs_dev = np.abs(values - mean_val)
        idx_worst = np.argmax(abs_dev)
        reduced = np.delete(values, idx_worst)
        mean_dw = np.mean(reduced)
        sd_dw = np.std(reduced, ddof=1)
        cv_dw = coefficient_of_variation(sd_dw, mean_dw)
        result["drop_worst"] = {
            "n": len(reduced),
            "mean": mean_dw,
            "sd": sd_dw,
            "cv": cv_dw,
            "removed_value": values[idx_worst],
        }

    return result
