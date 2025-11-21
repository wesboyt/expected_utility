import math
def calculate_sharpe_and_variance(params: dict) -> (float, float):
    ev = params['ev']
    risk = params['risk']
    ev_std = params['ev_std']
    risk_std = params['risk_std']

    if risk <= 0:
        return None, None
    sharpe_ratio = ev / risk
    var_s = (1 / risk) ** 2 * (ev_std ** 2) + (-ev / risk ** 2) ** 2 * (risk_std ** 2)

    return sharpe_ratio, var_s

def compare_option_efficiency(option_a: dict, option_b: dict, decision_hurdle_rate: float = 0.0) -> bool:
    s_a, var_s_a = calculate_sharpe_and_variance(option_a)
    s_b, var_s_b = calculate_sharpe_and_variance(option_b)

    if s_a is None or s_b is None:
        return False
    difference = (s_a - s_b) - decision_hurdle_rate
    std_err_diff = math.sqrt(var_s_a + var_s_b)

    if std_err_diff == 0:
        return (s_a - s_b) > decision_hurdle_rate

    z_stat = difference / std_err_diff
    return z_stat > 1.645
