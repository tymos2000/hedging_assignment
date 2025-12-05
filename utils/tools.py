import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

def bs_price(S, K, r, sigma, ttm):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * ttm) / (sigma * np.sqrt(ttm))
    d2 = d1 - sigma * np.sqrt(ttm)

    c = S * norm.cdf(d1) - K * np.exp(-r * ttm) * norm.cdf(d2)
    return c

def get_vega(S, K, r, sigma, ttm):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * ttm) / (sigma * np.sqrt(ttm))
    vega = S * norm.pdf(d1) * np.sqrt(ttm)
    return vega

def get_delta(S, K, r, sigma, ttm):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * ttm) / (sigma * np.sqrt(ttm))
    delta = norm.cdf(d1)
    return delta

def implied_vol_old(C, S, K, r, ttm, max_iterations=100):
    if ttm <= 0 or S <= 0 or K <=0 or C <= 0:
        return np.nan

    sigma = 0.2  # initial guess

    for _ in range(max_iterations):
        price = bs_price(S, K, r, sigma, ttm)
        
        diff = price - C

        if abs(diff) < 1e-8:
            return sigma
    
        vega = get_vega(S, K, r, sigma, ttm)
        sigma -= diff / vega

        if sigma <= 0:
            sigma *= 0.5

    return np.nan

def implied_vol(C, S, K, r, ttm):
    # basic sanity checks
    if ttm <= 0 or S <= 0 or K <= 0 or C <= 0:
        return np.nan

    # lower and upper bounds for sigma
    def f(sigma):
        return bs_price(S, K, r, sigma, ttm) - C

    try:
        # solve f(sigma) = 0
        return brentq(f, 1e-8, 10, maxiter=1000, xtol=1e-4)
    except ValueError:
        # no root in [1e-6, 5]
        return np.nan
    except RuntimeError:
        # numerical issues
        return np.nan