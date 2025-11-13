def calculate_sharpe_and_variance(params: dict) -> (float, float):
    """Calculates S = ev / risk and Var(S) via Delta Method."""
    ev = params['ev']
    risk = params['risk']
    ev_std = params['ev_std']
    risk_std = params['risk_std']

    # Handle division by zero for risk.
    # If risk is zero (or negative), efficiency is undefined or infinite.
    # We return (None, None) to signal an invalid comparison.
    if risk <= 0:
        print(f"Warning: Option with ev={ev} has non-positive risk={risk}. Cannot calculate efficiency.")
        return None, None

    # 1. Calculate the point estimate for the Sharpe Ratio
    sharpe_ratio = ev / risk

    # 2. Calculate the variance of the Sharpe Ratio using the Delta Method
    # Var(S) ≈ (∂S/∂ev)² * Var(ev) + (∂S/∂risk)² * Var(risk)
    # ∂S/∂ev = 1 / risk
    # ∂S/∂risk = -ev / risk²
    # Var(ev) = ev_std²
    # Var(risk) = risk_std²

    var_s = (1 / risk) ** 2 * (ev_std ** 2) + (-ev / risk ** 2) ** 2 * (risk_std ** 2)

    return sharpe_ratio, var_s

def compare_option_efficiency(option_a: dict, option_b: dict, decision_hurdle_rate: float = 0.0) -> bool:
    """
    Compares the efficiency (Sharpe Ratio) of two options using a
    statistical test to account for uncertainty in the estimates.

    This function interprets the inputs as:
    - ev: The point estimate of the expected value (mean).
    - ev_std: The standard error of the 'ev' estimate.
    - risk: The point estimate of the option's risk (standard deviation).
    - risk_std: The standard error of the 'risk' estimate.

    It assumes a risk-free rate of 0 for the Sharpe Ratio calculation (S = ev / risk).
    It then uses the Delta Method to estimate the variance of S and performs a
    Z-test to determine if Option A is statistically more efficient than Option B.

    Args:
        option_a: A dictionary with keys {'ev', 'ev_std', 'risk', 'risk_std'}.
        option_b: A dictionary with keys {'ev', 'ev_std', 'risk', 'risk_std'}.
        significance_level: The p-value threshold (e.g., 0.05 for 95%
                          confidence). The function tests if P(S_A > S_B).
        decision_hurdle_rate: An absolute premium required for S_A to be
                          considered better. The test becomes:
                          P(S_A > S_B + decision_hurdle_rate).
                          Your 15% value would be 0.15 here.

    Returns:
        True if Option A is statistically more efficient than Option B
        (by at least the hurdle rate) at the given significance level. False otherwise.
    """

    s_a, var_s_a = calculate_sharpe_and_variance(option_a)
    s_b, var_s_b = calculate_sharpe_and_variance(option_b)

    # If either calculation failed, we cannot compare.
    if s_a is None or s_b is None:
        return False

    # 3. Test the hypothesis H₁: S_A > S_B + hurdle (or S_A - S_B - hurdle > 0)

    # Calculate the Z-statistic for the difference adjusted by the hurdle rate.
    difference = (s_a - s_b) - decision_hurdle_rate

    # Standard error of the difference
    # SE_diff = sqrt(Var(S_A) + Var(S_B)) (assuming independent samples)
    std_err_diff = math.sqrt(var_s_a + var_s_b)

    # Handle division by zero if standard error is somehow zero
    if std_err_diff == 0:
        # If no variance, just compare point estimates against the hurdle.
        return (s_a - s_b) > decision_hurdle_rate

    z_stat = difference / std_err_diff

    # 4. Find the critical Z-value for a one-tailed test.
    # We are testing if S_A is *greater than* S_B.
                                
    z_critical_value = 1.645

    # Return true if our Z-statistic is greater than the critical value.
    # This means we are (1 - significance_level)% confident that S_A > S_B + hurdle.
    return z_stat > z_critical_value
