"""Mean-variance frontiers, with and without short sales.

Notation: `mu` is the vector of expected returns, `Sigma` is the covariance
matrix of returns, and `w` is a vector of portfolio weights that sums to one.

The frontier problem is

    minimize    w' Sigma w
    subject to  w' mu = target,   sum(w) = 1.

With only these two equality constraints, the problem has a closed-form
solution (`efficient_weights`). Add the no-short-sales constraint `w >= 0` and
it no longer does, so we solve it numerically (`minimize_variance`). The
notebooks `01_markowitz.ipynb.py` and `02_markowitz_derivation.ipynb.py`
explain why.
"""

import numpy as np
from scipy.optimize import minimize


def frontier_constants(mu, Sigma):
    """The scalars A, B, C, D that summarize the unconstrained frontier."""
    ones = np.ones(len(mu))
    Sigma_inv_ones = np.linalg.solve(Sigma, ones)
    Sigma_inv_mu = np.linalg.solve(Sigma, mu)
    A = ones @ Sigma_inv_ones
    B = ones @ Sigma_inv_mu
    C = mu @ Sigma_inv_mu
    D = A * C - B**2
    return A, B, C, D


def efficient_weights(mu, Sigma, target):
    """Closed-form minimum variance weights for a target mean (shorts allowed)."""
    ones = np.ones(len(mu))
    A, B, C, D = frontier_constants(mu, Sigma)
    Sigma_inv_ones = np.linalg.solve(Sigma, ones)
    Sigma_inv_mu = np.linalg.solve(Sigma, mu)
    return ((C - B * target) * Sigma_inv_ones + (A * target - B) * Sigma_inv_mu) / D


def frontier_variance(mu, Sigma, targets):
    """Closed-form minimum variance for each target mean (shorts allowed)."""
    A, B, C, D = frontier_constants(mu, Sigma)
    targets = np.asarray(targets)
    return (A * targets**2 - 2 * B * targets + C) / D


def minimize_variance(mu, Sigma, target=None, allow_short=True):
    """Numerically solve for the minimum variance weights.

    If `target` is None, there is no constraint on the mean, which gives the
    global minimum variance portfolio. If `allow_short` is False, every weight
    is constrained to be nonnegative.
    """
    n = len(mu)
    scale = 1 / np.mean(np.diag(Sigma))  # keeps the objective near 1 for the solver

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    x0 = np.ones(n) / n
    if target is not None:
        spread = mu.max() - mu.min()
        constraints.append({"type": "eq", "fun": lambda w: (w @ mu - target) / spread})
        # Start from a portfolio that already hits the target: a mix of the
        # lowest-mean and highest-mean assets.
        share = (target - mu.min()) / spread
        x0 = np.zeros(n)
        x0[mu.argmin()] += 1 - share
        x0[mu.argmax()] += share

    result = minimize(
        fun=lambda w: scale * w @ Sigma @ w,
        jac=lambda w: 2 * scale * Sigma @ w,
        x0=x0,
        bounds=None if allow_short else [(0, None)] * n,
        constraints=constraints,
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 500},
    )
    w = result.x
    # At the ends of the no-short frontier only one portfolio is feasible, and
    # the solver reports that it cannot improve. Accept any feasible answer.
    feasible = np.isclose(w.sum(), 1, atol=1e-8) and (
        target is None or np.isclose(w @ mu, target, atol=1e-8)
    )
    if not (result.success or feasible):
        raise RuntimeError(f"Optimizer failed: {result.message}")
    return w


def max_sharpe_weights(mu, Sigma, rf, allow_short=True):
    """Numerically solve for the portfolio with the highest Sharpe ratio.

    Uses a standard change of variables that turns the ratio into a quadratic
    program: minimize y' Sigma y subject to y' (mu - rf) = 1, then rescale y so
    that it sums to one.
    """
    n = len(mu)
    excess = mu - rf
    scale = 1 / np.mean(np.diag(Sigma))

    result = minimize(
        fun=lambda y: scale * y @ Sigma @ y,
        jac=lambda y: 2 * scale * Sigma @ y,
        x0=np.ones(n) / (np.ones(n) @ excess),
        bounds=None if allow_short else [(0, None)] * n,
        constraints=[{"type": "eq", "fun": lambda y: y @ excess - 1}],
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 500},
    )
    if not result.success:
        raise RuntimeError(f"Optimizer failed: {result.message}")
    return result.x / result.x.sum()


def frontier(mu, Sigma, targets, allow_short=True):
    """Minimum variance weights for each target mean, one row per target."""
    if allow_short:
        return np.array([efficient_weights(mu, Sigma, m) for m in targets])
    return np.array(
        [minimize_variance(mu, Sigma, m, allow_short=False) for m in targets]
    )
