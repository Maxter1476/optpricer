"""README figure: CRR convergence to Black-Scholes and the MC error budget."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from optpricer import bs_price, crr_price, mc_price

ARGS = (100.0, 110.0, 0.05, 0.25, 1.0)


def main() -> None:
    exact = bs_price(*ARGS)

    steps = np.unique(np.logspace(1, 3.3, 30).astype(int))
    tree_err = [abs(crr_price(*ARGS, steps=int(n)) - exact) for n in steps]

    paths = np.unique(np.logspace(3, 6, 12).astype(int))
    mc_err, mc_se = [], []
    for n in paths:
        result = mc_price(*ARGS, n_paths=int(n), seed=7)
        mc_err.append(abs(result.price - exact))
        mc_se.append(result.std_error)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.loglog(steps, tree_err, "o-", ms=4, label="CRR |error|")
    ax1.loglog(steps, tree_err[0] * steps[0] / steps, "k--", lw=1, label=r"$\propto 1/n$")
    ax1.set_xlabel("tree steps")
    ax1.set_ylabel("absolute error vs Black-Scholes")
    ax1.set_title("Binomial convergence")
    ax1.legend()

    ax2.loglog(paths, mc_err, "o", ms=4, label="MC |error|")
    ax2.loglog(paths, mc_se, "s--", ms=4, label="reported std error")
    ax2.loglog(
        paths, mc_se[0] * np.sqrt(paths[0] / paths), "k--", lw=1, label=r"$\propto 1/\sqrt{n}$"
    )
    ax2.set_xlabel("simulated paths")
    ax2.set_title("Monte Carlo error (antithetic + control variate)")
    ax2.legend()

    fig.suptitle("Three pricers, one answer — errors shrink at the theoretical rates")
    fig.tight_layout()
    fig.savefig("docs/figures/convergence.png", dpi=150)
    print("wrote docs/figures/convergence.png")


if __name__ == "__main__":
    main()
