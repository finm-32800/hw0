"""Interactive mean-variance dashboard.

Run it from the root of the repository with:

    streamlit run src/app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
import mean_variance

EXTRACT_PATH = config.DATA_DIR / "crsp_monthly_returns.csv"

st.set_page_config(page_title="Markowitz Portfolio Selection", layout="wide")


@st.cache_data
def load_returns():
    """Load the CRSP extract if it is there. Otherwise simulate returns, so the
    app still runs right after cloning the repository."""
    if EXTRACT_PATH.exists():
        df = pd.read_csv(EXTRACT_PATH, parse_dates=["date"], index_col="date")
        return df.drop(columns=["MKT", "RF"]), df["RF"].mean(), True

    rng = np.random.default_rng(0)
    names = [f"Stock {c}" for c in "ABCDEFGH"]
    market = rng.normal(0.008, 0.045, size=300)
    betas = rng.uniform(0.5, 1.5, size=len(names))
    noise = rng.normal(0.002, 0.06, size=(300, len(names)))
    dates = pd.date_range("2000-01-31", periods=300, freq="ME")
    simulated = pd.DataFrame(np.outer(market, betas) + noise, index=dates, columns=names)
    return simulated, 0.0015, False


@st.cache_data
def no_short_frontier(mu, Sigma, n_points=50):
    """The no-short frontier has no closed form, so it is solved point by point."""
    targets = np.linspace(mu.min(), mu.max(), n_points)
    weights = mean_variance.frontier(mu, Sigma, targets, allow_short=False)
    vols = np.sqrt([w @ Sigma @ w for w in weights])
    return targets, weights, vols


returns, rf, using_crsp = load_returns()

st.title("Portfolio Selection (Markowitz, 1952)")
if not using_crsp:
    st.warning(
        "Showing simulated returns because `_data/crsp_monthly_returns.csv` was "
        "not found. Run `doit` (or `python src/pull_crsp.py`) to download the data."
    )

tab_two, tab_many = st.tabs(["Two assets", "Many assets"])

with tab_two:
    left, right = st.columns([1, 3])
    with left:
        a = st.selectbox("Asset 1", returns.columns, index=0)
        b = st.selectbox("Asset 2", returns.columns, index=1)
        sample_rho = float(returns[a].corr(returns[b]))
        rho = st.slider("Correlation", -1.0, 1.0, round(sample_rho, 2), 0.01)
        w_a = st.slider(f"Weight on {a}", -0.5, 1.5, 0.5, 0.01)
        st.caption(f"The correlation in the data is {sample_rho:.2f}.")

    mu_a, mu_b = 12 * returns[a].mean(), 12 * returns[b].mean()
    sd_a, sd_b = np.sqrt(12) * returns[a].std(), np.sqrt(12) * returns[b].std()

    def two_asset(w):
        mean = w * mu_a + (1 - w) * mu_b
        var = w**2 * sd_a**2 + (1 - w) ** 2 * sd_b**2 + 2 * w * (1 - w) * rho * sd_a * sd_b
        return mean, np.sqrt(np.maximum(var, 0))

    grid = np.linspace(-0.5, 1.5, 201)
    mean_grid, sd_grid = two_asset(grid)
    mean_now, sd_now = two_asset(w_a)

    with right:
        col1, col2, col3 = st.columns(3)
        col1.metric("Expected return", f"{mean_now:.1%}")
        col2.metric("Volatility", f"{sd_now:.1%}")
        col3.metric(
            "Weighted average of the two volatilities",
            f"{abs(w_a) * sd_a + abs(1 - w_a) * sd_b:.1%}",
        )

        fig = go.Figure()
        fig.add_scatter(x=sd_grid, y=mean_grid, mode="lines", name="All mixes of the two")
        fig.add_scatter(
            x=[sd_a, sd_b], y=[mu_a, mu_b], mode="markers+text", text=[a, b],
            textposition="top center", marker=dict(size=10), name="Assets",
        )
        fig.add_scatter(
            x=[sd_now], y=[mean_now], mode="markers",
            marker=dict(size=16, symbol="star"), name="Your portfolio",
        )
        fig.update_layout(
            xaxis_title="Volatility (annualized)", yaxis_title="Expected return (annualized)",
            xaxis_tickformat=".0%", yaxis_tickformat=".0%", height=480,
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "The expected return moves in a straight line as you change the weight. "
            "The volatility does not. Drag the correlation toward -1 and watch the curve bend."
        )

with tab_many:
    left, right = st.columns([1, 3])
    with left:
        chosen = st.multiselect("Assets", list(returns.columns), default=list(returns.columns))
        years = sorted(returns.index.year.unique())
        start, end = st.select_slider(
            "Estimation window", options=years, value=(years[0], years[-1])
        )
        no_shorts = st.checkbox("No short sales (weights cannot be negative)")
        st.caption(
            "Shorten or shift the estimation window and watch the tangency weights. "
            "The minimum variance weights move much less. With shorts allowed, the "
            "frontier comes from a formula. With no short sales there is no formula, "
            "so each point is solved numerically."
        )

    sample = returns.loc[str(start) : str(end), chosen]
    if len(chosen) < 2 or len(sample) <= len(chosen):
        st.info("Choose at least two assets and a window with more months than assets.")
    else:
        mu = sample.mean().values
        Sigma = sample.cov().values

        def stats(w):
            return 12 * w @ mu, np.sqrt(12 * w @ Sigma @ w)

        # Unconstrained frontier, from the closed form.
        A, B, _, _ = mean_variance.frontier_constants(mu, Sigma)
        top = max(mu.max(), B / A) * 1.6
        targets = np.linspace(min(mu.min(), 0), top, 200)
        vols = np.sqrt(mean_variance.frontier_variance(mu, Sigma, targets))

        w_gmv = mean_variance.minimize_variance(mu, Sigma, None, allow_short=not no_shorts)
        try:
            w_tan = mean_variance.max_sharpe_weights(mu, Sigma, rf, allow_short=not no_shorts)
        except RuntimeError:
            w_tan = None  # e.g. no asset beat the risk-free rate in this window

        with right:
            fig = go.Figure()
            fig.add_scatter(
                x=np.sqrt(12) * vols, y=12 * targets, mode="lines", name="Frontier, shorts allowed",
                line=dict(dash="dot" if no_shorts else "solid"),
            )
            if no_shorts:
                targets_ns, weights_ns, vols_ns = no_short_frontier(mu, Sigma)
                fig.add_scatter(
                    x=np.sqrt(12) * vols_ns, y=12 * targets_ns, mode="lines",
                    name="Frontier, no shorts", line=dict(width=4),
                )
            fig.add_scatter(
                x=np.sqrt(12) * sample.std(), y=12 * sample.mean(), mode="markers+text",
                text=chosen, textposition="top center", name="Assets",
                marker=dict(color="gray"),
            )
            mean_gmv, sd_gmv = stats(w_gmv)
            fig.add_scatter(
                x=[sd_gmv], y=[mean_gmv], mode="markers",
                marker=dict(size=14, symbol="diamond"), name="Minimum variance",
            )
            if w_tan is not None:
                mean_tan, sd_tan = stats(w_tan)
                fig.add_scatter(
                    x=[sd_tan], y=[mean_tan], mode="markers",
                    marker=dict(size=16, symbol="star"), name="Tangency",
                )
            fig.update_layout(
                xaxis_title="Volatility (annualized)", yaxis_title="Expected return (annualized)",
                xaxis_tickformat=".0%", yaxis_tickformat=".0%", height=520,
                xaxis_range=[0, float(np.sqrt(12) * sample.std().max()) * 1.1],
            )
            st.plotly_chart(fig, width="stretch")

            weights = pd.DataFrame({"Minimum variance": w_gmv}, index=chosen)
            if w_tan is not None:
                weights.insert(0, "Tangency", w_tan)
            else:
                st.info("No tangency portfolio: no asset beat the risk-free rate in this window.")
            bars = go.Figure()
            for name in weights.columns:
                bars.add_bar(x=weights.index, y=weights[name], name=name)
            bars.update_layout(
                barmode="group", yaxis_tickformat=".0%", yaxis_title="Portfolio weight", height=320
            )
            st.plotly_chart(bars, width="stretch")

            if no_shorts:
                efficient = targets_ns >= weights_ns[np.argmin(vols_ns)] @ mu
                area = go.Figure()
                for i, name in enumerate(chosen):
                    area.add_scatter(
                        x=12 * targets_ns[efficient], y=weights_ns[efficient, i],
                        mode="lines", stackgroup="one", name=name,
                    )
                area.update_layout(
                    title="Composition of the efficient no-short frontier",
                    xaxis_title="Target expected return (annualized)",
                    yaxis_title="Portfolio weight",
                    xaxis_tickformat=".0%", yaxis_tickformat=".0%", height=380,
                )
                st.plotly_chart(area, width="stretch")
