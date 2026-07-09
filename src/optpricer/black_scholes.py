"""Analytic Black-Scholes-Merton prices and Greeks for European options.

Supports a continuous dividend yield q. All Greeks are closed-form and the
test suite verifies each one against central finite differences of the price
— the analytic formulas and the numerics must agree independently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

__all__ = ["OptionQuote", "bs_price", "bs_greeks"]


@dataclass(frozen=True)
class OptionQuote:
    """Greeks bundle returned by :func:`bs_greeks`."""

    price: float
    delta: float
    gamma: float
    vega: float  # per 1.0 of vol (not per %)
    theta: float  # per year (not per day)
    rho: float  # per 1.0 of rate


def _d1_d2(spot: float, strike: float, rate: float, vol: float, tau: float, div: float):
    sqrt_tau = np.sqrt(tau)
    d1 = (np.log(spot / strike) + (rate - div + 0.5 * vol**2) * tau) / (vol * sqrt_tau)
    return d1, d1 - vol * sqrt_tau


def _validate(spot: float, strike: float, vol: float, tau: float) -> None:
    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive")
    if vol <= 0:
        raise ValueError("volatility must be positive")
    if tau < 0:
        raise ValueError("time to expiry must be non-negative")


def bs_price(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    tau: float,
    *,
    kind: str = "call",
    div: float = 0.0,
) -> float:
    """Black-Scholes-Merton price of a European call or put.

    ``tau`` is time to expiry in years; ``div`` a continuous dividend yield.
    At tau = 0 returns intrinsic value.
    """
    _validate(spot, strike, vol, tau)
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")
    if tau == 0.0:
        intrinsic = spot - strike if kind == "call" else strike - spot
        return max(intrinsic, 0.0)
    d1, d2 = _d1_d2(spot, strike, rate, vol, tau, div)
    df_spot = spot * np.exp(-div * tau)
    df_strike = strike * np.exp(-rate * tau)
    if kind == "call":
        return float(df_spot * norm.cdf(d1) - df_strike * norm.cdf(d2))
    return float(df_strike * norm.cdf(-d2) - df_spot * norm.cdf(-d1))


def bs_greeks(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    tau: float,
    *,
    kind: str = "call",
    div: float = 0.0,
) -> OptionQuote:
    """Closed-form Greeks (delta, gamma, vega, theta, rho) plus the price."""
    _validate(spot, strike, vol, tau)
    if tau == 0.0:
        raise ValueError("Greeks are undefined at expiry; pass tau > 0")
    d1, d2 = _d1_d2(spot, strike, rate, vol, tau, div)
    sqrt_tau = np.sqrt(tau)
    disc_r = np.exp(-rate * tau)
    disc_q = np.exp(-div * tau)
    pdf_d1 = norm.pdf(d1)

    gamma = disc_q * pdf_d1 / (spot * vol * sqrt_tau)
    vega = spot * disc_q * pdf_d1 * sqrt_tau
    common_theta = -spot * disc_q * pdf_d1 * vol / (2.0 * sqrt_tau)
    if kind == "call":
        delta = disc_q * norm.cdf(d1)
        theta = (
            common_theta
            - rate * strike * disc_r * norm.cdf(d2)
            + div * spot * disc_q * norm.cdf(d1)
        )
        rho = strike * tau * disc_r * norm.cdf(d2)
    elif kind == "put":
        delta = -disc_q * norm.cdf(-d1)
        theta = (
            common_theta
            + rate * strike * disc_r * norm.cdf(-d2)
            - div * spot * disc_q * norm.cdf(-d1)
        )
        rho = -strike * tau * disc_r * norm.cdf(-d2)
    else:
        raise ValueError("kind must be 'call' or 'put'")
    price = bs_price(spot, strike, rate, vol, tau, kind=kind, div=div)
    return OptionQuote(
        price=price,
        delta=float(delta),
        gamma=float(gamma),
        vega=float(vega),
        theta=float(theta),
        rho=float(rho),
    )
