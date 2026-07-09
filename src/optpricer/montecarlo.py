"""Monte Carlo pricing under geometric Brownian motion.

Variance reduction: antithetic variates plus a control variate (the terminal
spot, whose discounted risk-neutral expectation is known exactly). The test
suite requires the estimator's reported standard error to be honest — the
true Black-Scholes price must fall inside +-3 SE — and that the control
variate demonstrably shrinks the error bar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["MCResult", "mc_price"]


@dataclass(frozen=True)
class MCResult:
    price: float
    std_error: float
    n_paths: int


def mc_price(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    tau: float,
    *,
    kind: str = "call",
    div: float = 0.0,
    n_paths: int = 100_000,
    antithetic: bool = True,
    control_variate: bool = True,
    seed: int | None = None,
) -> MCResult:
    """European option price by GBM terminal-value simulation."""
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")
    if n_paths < 100:
        raise ValueError("n_paths must be >= 100")
    rng = np.random.default_rng(seed)

    half = n_paths // 2
    z = rng.standard_normal(half if antithetic else n_paths)
    if antithetic:
        z = np.concatenate([z, -z])

    drift = (rate - div - 0.5 * vol**2) * tau
    terminal = spot * np.exp(drift + vol * np.sqrt(tau) * z)
    if kind == "call":
        payoff = np.maximum(terminal - strike, 0.0)
    else:
        payoff = np.maximum(strike - terminal, 0.0)
    discounted = np.exp(-rate * tau) * payoff

    if control_variate:
        # Control: discounted terminal spot, E[.] = spot * exp(-div tau) exactly.
        control = np.exp(-rate * tau) * terminal
        control_mean = spot * np.exp(-div * tau)
        cov = np.cov(discounted, control, ddof=1)
        beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0.0
        discounted = discounted - beta * (control - control_mean)

    n = len(discounted)
    return MCResult(
        price=float(discounted.mean()),
        std_error=float(discounted.std(ddof=1) / np.sqrt(n)),
        n_paths=n,
    )
