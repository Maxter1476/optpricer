import numpy as np
import pytest

from optpricer import bs_greeks, bs_price


def test_hull_textbook_value():
    """Hull's classic example: S=42, K=40, r=10%, sigma=20%, T=0.5.
    Published: call = 4.76, put = 0.81."""
    call = bs_price(42, 40, 0.10, 0.20, 0.5, kind="call")
    put = bs_price(42, 40, 0.10, 0.20, 0.5, kind="put")
    assert call == pytest.approx(4.759, abs=2e-3)
    assert put == pytest.approx(0.808, abs=2e-3)


def test_atm_one_year_benchmark():
    """S=K=100, r=5%, sigma=20%, T=1: call = 10.4506 (standard benchmark)."""
    assert bs_price(100, 100, 0.05, 0.2, 1.0) == pytest.approx(10.4506, abs=1e-4)


@pytest.mark.parametrize("spot", [60.0, 95.0, 100.0, 140.0])
@pytest.mark.parametrize("div", [0.0, 0.03])
def test_put_call_parity(spot, div):
    """C - P = S e^{-q tau} - K e^{-r tau}, an arbitrage identity."""
    strike, rate, vol, tau = 100.0, 0.04, 0.25, 0.75
    call = bs_price(spot, strike, rate, vol, tau, kind="call", div=div)
    put = bs_price(spot, strike, rate, vol, tau, kind="put", div=div)
    parity = spot * np.exp(-div * tau) - strike * np.exp(-rate * tau)
    assert call - put == pytest.approx(parity, abs=1e-12)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("div", [0.0, 0.02])
def test_greeks_match_finite_differences(kind, div):
    spot, strike, rate, vol, tau = 105.0, 100.0, 0.05, 0.3, 0.8
    quote = bs_greeks(spot, strike, rate, vol, tau, kind=kind, div=div)

    eps = 1e-4

    def price(s=spot, r=rate, v=vol, t=tau):
        return bs_price(s, strike, r, v, t, kind=kind, div=div)

    delta_fd = (price(s=spot + eps) - price(s=spot - eps)) / (2 * eps)
    gamma_fd = (price(s=spot + eps) - 2 * price() + price(s=spot - eps)) / eps**2
    vega_fd = (price(v=vol + eps) - price(v=vol - eps)) / (2 * eps)
    theta_fd = -(price(t=tau + eps) - price(t=tau - eps)) / (2 * eps)
    rho_fd = (price(r=rate + eps) - price(r=rate - eps)) / (2 * eps)

    assert quote.delta == pytest.approx(delta_fd, abs=1e-6)
    assert quote.gamma == pytest.approx(gamma_fd, abs=1e-4)
    assert quote.vega == pytest.approx(vega_fd, abs=1e-4)
    assert quote.theta == pytest.approx(theta_fd, abs=1e-4)
    assert quote.rho == pytest.approx(rho_fd, abs=1e-4)


def test_intrinsic_at_expiry():
    assert bs_price(120, 100, 0.05, 0.2, 0.0) == pytest.approx(20.0)
    assert bs_price(80, 100, 0.05, 0.2, 0.0, kind="put") == pytest.approx(20.0)


def test_monotone_in_vol():
    prices = [bs_price(100, 100, 0.03, v, 1.0) for v in (0.1, 0.2, 0.4, 0.8)]
    assert all(a < b for a, b in zip(prices, prices[1:], strict=False))


def test_input_validation():
    with pytest.raises(ValueError):
        bs_price(-1, 100, 0.05, 0.2, 1.0)
    with pytest.raises(ValueError):
        bs_price(100, 100, 0.05, -0.2, 1.0)
    with pytest.raises(ValueError):
        bs_price(100, 100, 0.05, 0.2, 1.0, kind="straddle")
