"""Implied volatility inversion: Newton with a bisection safeguard."""

from __future__ import annotations

import numpy as np

from .black_scholes import bs_greeks

__all__ = ["implied_vol"]


def implied_vol(
    target_price: float,
    spot: float,
    strike: float,
    rate: float,
    tau: float,
    *,
    kind: str = "call",
    div: float = 0.0,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> float:
    """Volatility such that the Black-Scholes price equals ``target_price``.

    Rejects prices outside the no-arbitrage band before iterating. Newton
    steps use the analytic vega; whenever a step leaves the current bracket,
    it falls back to bisection, so convergence is guaranteed.
    """
    if tau <= 0:
        raise ValueError("tau must be positive")
    disc_spot = spot * np.exp(-div * tau)
    disc_strike = strike * np.exp(-rate * tau)
    if kind == "call":
        lower, upper = max(disc_spot - disc_strike, 0.0), disc_spot
    elif kind == "put":
        lower, upper = max(disc_strike - disc_spot, 0.0), disc_strike
    else:
        raise ValueError("kind must be 'call' or 'put'")
    if not lower <= target_price <= upper:
        raise ValueError(
            f"price {target_price} violates no-arbitrage bounds [{lower:.6g}, {upper:.6g}]"
        )

    vol_low, vol_high = 1e-6, 5.0
    vol = 0.3
    for _ in range(max_iter):
        quote = bs_greeks(spot, strike, rate, vol, tau, kind=kind, div=div)
        diff = quote.price - target_price
        if abs(diff) < tol:
            return vol
        if diff > 0:
            vol_high = vol
        else:
            vol_low = vol
        newton = vol - diff / quote.vega if quote.vega > 1e-12 else np.nan
        vol = newton if np.isfinite(newton) and vol_low < newton < vol_high else 0.5 * (
            vol_low + vol_high
        )
    raise RuntimeError("implied vol did not converge")
