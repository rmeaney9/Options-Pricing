import numpy as np
from scipy.stats import norm

def bs_inputs(S, K, T, r, sigma):
    """
    Compute d1 and d2 for the Black-Scholes formula.

    Parameters:
    S : float or np.ndarray
        Spot price(s)
    K : float
        Strike price
    T : float
        Time to maturity (in years)
    r : float
        Risk-free rate (annualised)
    sigma : float or np.ndarray
        Volatility (annualised)

    Returns:
    d1, d2 : np.ndarray or float
    """
    S = np.array(S)
    sigma = np.array(sigma)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2

def call_price(S, K, T, r, sigma, option_type="call"):
    """
    Calculate the Black-Scholes price for a European call or put option.

    Parameters:
    S : float or np.ndarray
        Spot price(s)
    K : float
        Strike price
    T : float
        Time to maturity (in years)
    r : float
        Risk-free interest rate (annualised)
    sigma : float or np.ndarray
        Volatility (annualised)
    option_type : str
        "call" or "put"

    Returns:
    np.ndarray or float : option price(s)
    """
    d1, d2 = bs_inputs(S, K, T, r, sigma)

    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")