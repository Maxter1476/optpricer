"""The heart of the package: three independent pricing methods must agree."""

import numpy as np
import pytest

from optpricer import bs_price, crr_price, implied_vol, mc_price


def test_binomial_converges_to_black_scholes_at_rate_1_over_n():
    """CRR error should shrink ~O(1/n): quadrupling steps cuts error ~4x."""
    args = (100.0, 110.0, 0.05, 0.25, 1.0)
    exact = bs_price(*args)
    err = {n: abs(crr_price(*args, steps=n) - exact) for n in (50, 200, 800)}
    assert err[800] < 1e-2
    # allow slack for the oscillatory CRR convergence; the trend must be ~1/n
    assert err[200] < err[50]
    assert err[800] < err[200]
    assert err[800] < err[50] / 4.0


@pytest.mark.parametrize("kind", ["call", "put"])
def test_monte_carlo_within_3_sigma(kind):
    args = dict(spot=100.0, strike=95.0, rate=0.04, vol=0.3, tau=0.5, kind=kind)
    exact = bs_price(
        args["spot"], args["strike"], args["rate"], args["vol"], args["tau"], kind=kind
    )
    result = mc_price(**args, n_paths=200_000, seed=42)
    assert abs(result.price - exact) < 3.0 * result.std_error
    assert result.std_error < 0.05


def test_control_variate_reduces_error():
    args = dict(spot=100.0, strike=100.0, rate=0.05, vol=0.2, tau=1.0)
    plain = mc_price(**args, n_paths=100_000, control_variate=False, seed=1)
    controlled = mc_price(**args, n_paths=100_000, control_variate=True, seed=1)
    assert controlled.std_error < 0.5 * plain.std_error


def test_american_put_exceeds_european():
    """Early exercise on a deep ITM put has positive value with r > 0."""
    european = crr_price(80.0, 100.0, 0.08, 0.2, 1.0, kind="put", steps=800)
    american = crr_price(80.0, 100.0, 0.08, 0.2, 1.0, kind="put", style="american", steps=800)
    assert american > european + 0.05
    # and it can never be worth less than intrinsic
    assert american >= 20.0


def test_american_call_no_dividends_equals_european():
    """Merton: never exercise an American call early without dividends."""
    european = crr_price(105.0, 100.0, 0.05, 0.25, 1.0, kind="call", steps=800)
    american = crr_price(
        105.0, 100.0, 0.05, 0.25, 1.0, kind="call", style="american", steps=800
    )
    assert american == pytest.approx(european, abs=1e-9)


@pytest.mark.parametrize("true_vol", [0.08, 0.2, 0.55, 1.2])
@pytest.mark.parametrize("kind", ["call", "put"])
def test_implied_vol_round_trip(true_vol, kind):
    price = bs_price(100.0, 105.0, 0.03, true_vol, 0.6, kind=kind)
    recovered = implied_vol(price, 100.0, 105.0, 0.03, 0.6, kind=kind)
    assert recovered == pytest.approx(true_vol, abs=1e-8)


def test_implied_vol_rejects_arbitrage_prices():
    with pytest.raises(ValueError):
        implied_vol(200.0, 100.0, 105.0, 0.03, 0.6)  # above spot
    with pytest.raises(ValueError):
        implied_vol(0.0, 100.0, 50.0, 0.03, 0.6)  # below intrinsic floor


def test_tree_rejects_degenerate_probability():
    # huge drift + tiny vol + coarse steps -> p outside (0,1)
    with pytest.raises(ValueError):
        crr_price(100.0, 100.0, 2.5, 0.01, 1.0, steps=2)


def test_all_three_methods_agree():
    spot, strike, rate, vol, tau = 100.0, 103.0, 0.045, 0.28, 0.9
    analytic = bs_price(spot, strike, rate, vol, tau)
    tree = crr_price(spot, strike, rate, vol, tau, steps=2000)
    mc = mc_price(spot, strike, rate, vol, tau, n_paths=400_000, seed=9)
    assert tree == pytest.approx(analytic, abs=5e-3)
    assert abs(mc.price - analytic) < 3.0 * mc.std_error
    assert np.isfinite(mc.std_error)
