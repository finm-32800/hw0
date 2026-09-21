# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Appendix: Deriving the Mean-Variance Frontier
#
# This appendix derives the formulas used in the
# portfolio selection notebook, `01_markowitz`. The derivation follows
# Merton (1972). At the end, we check every formula numerically.
#
# ## Setup
#
# There are $N$ risky assets with expected returns $\mu$ (an $N$-vector) and
# covariance matrix $\Sigma$ ($N \times N$). We assume that $\Sigma$ is positive
# definite, so no asset is redundant, and that the assets do not all have the
# same expected return. A portfolio is a weight vector $w$ with
# $w^\top \mathbf{1} = 1$. Its expected return is $w^\top \mu$ and its variance
# is $w^\top \Sigma w$.
#
# ## The frontier problem
#
# For a target expected return $m$,
#
# $$
# \min_{w} \; \tfrac{1}{2} w^\top \Sigma w
# \quad \text{subject to} \quad
# w^\top \mu = m, \qquad w^\top \mathbf{1} = 1 .
# $$
#
# The factor of $\tfrac{1}{2}$ only tidies the derivative. The objective is
# strictly convex and the constraints are linear, so the first-order conditions
# identify the unique minimum.
#
# ## First-order conditions
#
# Form the Lagrangian with multipliers $\lambda$ and $\gamma$:
#
# $$
# \mathcal{L} = \tfrac{1}{2} w^\top \Sigma w
#   - \lambda \, (w^\top \mu - m) - \gamma \, (w^\top \mathbf{1} - 1).
# $$
#
# Differentiating with respect to $w$ and setting the result to zero,
#
# $$
# \Sigma w = \lambda \mu + \gamma \mathbf{1}
# \quad \Longrightarrow \quad
# w = \lambda \, \Sigma^{-1} \mu + \gamma \, \Sigma^{-1} \mathbf{1}. \tag{1}
# $$
#
# Every frontier portfolio is a combination of the same two vectors,
# $\Sigma^{-1} \mu$ and $\Sigma^{-1} \mathbf{1}$. Only the multipliers depend on
# the target $m$.
#
# ## Solving for the multipliers
#
# Define
#
# $$
# A = \mathbf{1}^\top \Sigma^{-1} \mathbf{1}, \quad
# B = \mathbf{1}^\top \Sigma^{-1} \mu, \quad
# C = \mu^\top \Sigma^{-1} \mu, \quad
# D = AC - B^2 .
# $$
#
# Multiply (1) on the left by $\mu^\top$ and then by $\mathbf{1}^\top$, and use
# the two constraints:
#
# $$
# \lambda C + \gamma B = m, \qquad \lambda B + \gamma A = 1 .
# $$
#
# This is a $2 \times 2$ linear system. Its solution is
#
# $$
# \lambda = \frac{A m - B}{D}, \qquad \gamma = \frac{C - B m}{D}. \tag{2}
# $$
#
# We need $D \neq 0$. In fact $D > 0$: because $\Sigma^{-1}$ is positive
# definite, the Cauchy-Schwarz inequality gives $B^2 \le AC$, with equality only
# if $\mu$ is proportional to $\mathbf{1}$, which we ruled out. Substituting (2)
# into (1) gives the optimal weights,
#
# $$
# w^*(m) = \frac{(C - Bm)\, \Sigma^{-1} \mathbf{1} + (Am - B)\, \Sigma^{-1} \mu}{D}.
# $$
#
# ## The frontier is a hyperbola
#
# Multiply the first-order condition $\Sigma w = \lambda \mu + \gamma \mathbf{1}$
# on the left by $w^\top$ and use the constraints again:
#
# $$
# \sigma^2(m) = w^\top \Sigma w = \lambda m + \gamma
#   = \frac{A m^2 - 2 B m + C}{D}.
# $$
#
# The variance is a parabola in $m$. In the usual picture, with the standard
# deviation $\sigma$ on the horizontal axis and $m$ on the vertical axis, the
# curve is a hyperbola.
#
# ## The global minimum variance portfolio
#
# Minimize $\sigma^2(m)$ over $m$: $2Am - 2B = 0$, so $m_{\text{GMV}} = B/A$.
# Then $\lambda = 0$ and $\gamma = 1/A$, and
#
# $$
# w_{\text{GMV}} = \frac{\Sigma^{-1} \mathbf{1}}{A}, \qquad
# \sigma^2_{\text{GMV}} = \frac{1}{A}.
# $$
#
# With $\lambda = 0$, the constraint on the mean does not bind, which is why
# $\mu$ drops out of the weights. The portfolios on the frontier with
# $m \ge B/A$ are the **efficient** ones. Those below have the same variance as
# an efficient portfolio and a lower mean.
#
# ## The two-fund theorem
#
# Define a second portfolio, $w_d = \Sigma^{-1} \mu / B$ (assuming $B \ne 0$).
# Then (1) can be written as
#
# $$
# w^*(m) = (\gamma A) \, w_{\text{GMV}} + (\lambda B) \, w_d ,
# \qquad \gamma A + \lambda B = 1 .
# $$
#
# Every frontier portfolio is a mix of the same two funds. This is why the main
# notebook could trace the frontier by mixing two portfolios, and it is the
# first hint of why, in equilibrium, all investors might hold the same risky
# portfolio.
#
# ## The tangency portfolio
#
# Add a risk-free asset with return $r_f < B/A$. An investor who mixes the
# risk-free asset with a risky portfolio $w$ moves along a straight line in
# mean-standard deviation space whose slope is the Sharpe ratio of $w$. The best
# line is the steepest one:
#
# $$
# \max_{w} \; \frac{w^\top \mu - r_f}{\sqrt{w^\top \Sigma w}}
# \quad \text{subject to} \quad w^\top \mathbf{1} = 1 .
# $$
#
# The ratio does not change when $w$ is scaled, so we can drop the constraint,
# solve, and rescale afterward. Setting the derivative of the ratio to zero
# gives $\Sigma w \propto \mu - r_f \mathbf{1}$. Rescaling so that the weights
# sum to one,
#
# $$
# w_{\text{tan}} = \frac{\Sigma^{-1} (\mu - r_f \mathbf{1})}{B - A r_f}.
# $$
#
# The condition $r_f < B/A$ says that the risk-free rate is below the expected
# return on the minimum variance portfolio. It guarantees that the denominator
# is positive and that the tangency point lies on the efficient half of the
# frontier.
#
# ## Why the no-short problem has no closed form
#
# Add the constraint $w \ge 0$, with a vector of multipliers $\nu$. The
# Karush-Kuhn-Tucker conditions are
#
# $$
# \Sigma w = \lambda \mu + \gamma \mathbf{1} + \nu, \qquad
# \nu \ge 0, \qquad w \ge 0, \qquad \nu_i w_i = 0 \text{ for every } i .
# $$
#
# The last condition, complementary slackness, says that each asset is either
# held ($w_i > 0$ and $\nu_i = 0$) or excluded ($w_i = 0$). *Given* the set of
# held assets, the conditions reduce to the unconstrained problem on that
# subset, and the formula above applies to it. The difficulty is finding the
# set. There are $2^N - 1$ candidates, and the right one changes with $m$.
#
# Markowitz (1956) showed that the set changes only at a finite number of
# "corner portfolios" and that, between corners, the weights are linear in $m$.
# His critical line algorithm walks from corner to corner. So the no-short
# frontier is a chain of hyperbola segments, and it is computed by an algorithm
# and not by a formula. We use a general-purpose numerical solver instead, since
# the problem is a small quadratic program.
#
# ## Checking the formulas numerically
#
# A derivation is a claim, and claims should be tested. Using the data from the
# main notebook, we compare each formula with a numerical optimizer that knows
# nothing about the algebra.

# %%
import numpy as np

import mean_variance
import pull_crsp

df = pull_crsp.load_extract()
stocks = df.drop(columns=["MKT", "RF"])
mu, Sigma, rf = stocks.mean().values, stocks.cov().values, df["RF"].mean()

A, B, C, D = mean_variance.frontier_constants(mu, Sigma)
print(f"A = {A:.1f}, B = {B:.3f}, C = {C:.5f}, D = {D:.4f}")
assert D > 0

# %%
target = 0.015  # a 1.5% monthly expected return

w_formula = mean_variance.efficient_weights(mu, Sigma, target)
w_numerical = mean_variance.minimize_variance(mu, Sigma, target, allow_short=True)

print("Largest difference in weights:", np.abs(w_formula - w_numerical).max())
print("Variance from the weights:    ", w_formula @ Sigma @ w_formula)
print("Variance from the formula:    ", mean_variance.frontier_variance(mu, Sigma, target))
assert np.allclose(w_formula, w_numerical, atol=1e-5)

# %%
w_gmv_formula = np.linalg.solve(Sigma, np.ones(len(mu))) / A
w_gmv_numerical = mean_variance.minimize_variance(mu, Sigma, target=None)

print("GMV mean, B/A:     ", B / A, "vs", w_gmv_numerical @ mu)
print("GMV variance, 1/A: ", 1 / A, "vs", w_gmv_numerical @ Sigma @ w_gmv_numerical)
assert np.allclose(w_gmv_formula, w_gmv_numerical, atol=1e-5)

# %%
w_tan_formula = np.linalg.solve(Sigma, mu - rf) / (B - A * rf)
w_tan_numerical = mean_variance.max_sharpe_weights(mu, Sigma, rf)

print("Tangency weights sum to:", w_tan_formula.sum())
assert rf < B / A
assert np.allclose(w_tan_formula, w_tan_numerical, atol=1e-5)

# %% [markdown]
# Finally, the claim about the no-short problem: given the set of assets that
# the constrained solution holds, the unconstrained formula applied to that
# subset reproduces the constrained weights.

# %%
w_no_short = mean_variance.minimize_variance(mu, Sigma, target, allow_short=False)
held = w_no_short > 1e-8

w_subset = mean_variance.efficient_weights(mu[held], Sigma[np.ix_(held, held)], target)

print("Assets held:", list(stocks.columns[held]))
print("Largest difference:", np.abs(w_no_short[held] - w_subset).max())
assert np.allclose(w_no_short[held], w_subset, atol=1e-5)

# %% [markdown]
# ## References
#
# - Markowitz, Harry. "The Optimization of a Quadratic Function Subject to
#   Linear Constraints." *Naval Research Logistics Quarterly* 3, no. 1-2 (1956):
#   111-133.
# - Merton, Robert C. "An Analytic Derivation of the Efficient Portfolio
#   Frontier." *Journal of Financial and Quantitative Analysis* 7, no. 4 (1972):
#   1851-1872.
