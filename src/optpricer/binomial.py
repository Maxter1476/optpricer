"""Cox-Ross-Rubinstein binomial trees, European and American exercise.

The backward induction is vectorized over tree levels (each level is one
NumPy array), so a 2,000-step tree prices in milliseconds. European CRR
prices must converge to Black-Scholes at O(1/n) — the test suite checks the
convergence rate, not just closeness.
"""

from __future__ import annotations

import numpy as np

__all__ = ["crr_price"]


def crr_price(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    tau: float,
    *,
    kind: str = "call",
    style: str = "european",
    steps: int = 500,
    div: float = 0.0,
) -> float:
    """Binomial (CRR) option price.

    ``style="american"`` enables early exercise via the standard comparison
    of continuation value against intrinsic value at every node.
    """
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")
    if style not in ("european", "american"):
        raise ValueError("style must be 'european' or 'american'")
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if tau <= 0:
        intrinsic = spot - strike if kind == "call" else strike - spot
        return max(intrinsic, 0.0)

    dt = tau / steps
    up = np.exp(vol * np.sqrt(dt))
    down = 1.0 / up
    growth = np.exp((rate - div) * dt)
    p_up = (growth - down) / (up - down)
    if not 0.0 < p_up < 1.0:
        raise ValueError(
            f"risk-neutral probability {p_up:.4f} outside (0,1); increase steps "
            "or check parameters (arbitrage in the discretized tree)"
        )
    discount = np.exp(-rate * dt)

    # Terminal prices S * u^j * d^(n-j), j = 0..n
    j = np.arange(steps + 1)
    prices = spot * up**j * down ** (steps - j)
    if kind == "call":
        payoff = np.maximum(prices - strike, 0.0)
    else:
        payoff = np.maximum(strike - prices, 0.0)

    for level in range(steps - 1, -1, -1):
        payoff = discount * (p_up * payoff[1:] + (1.0 - p_up) * payoff[:-1])
        if style == "american":
            j = np.arange(level + 1)
            prices = spot * up**j * down ** (level - j)
            intrinsic = (
                np.maximum(prices - strike, 0.0)
                if kind == "call"
                else np.maximum(strike - prices, 0.0)
            )
            payoff = np.maximum(payoff, intrinsic)
    return float(payoff[0])
