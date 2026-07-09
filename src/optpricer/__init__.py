"""optpricer — option pricing three ways, each validating the others."""

from .binomial import crr_price
from .black_scholes import OptionQuote, bs_greeks, bs_price
from .implied import implied_vol
from .montecarlo import MCResult, mc_price

__all__ = [
    "MCResult",
    "OptionQuote",
    "bs_greeks",
    "bs_price",
    "crr_price",
    "implied_vol",
    "mc_price",
]

__version__ = "0.1.0"
