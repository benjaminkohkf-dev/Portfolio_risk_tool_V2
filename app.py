"""
Factor Risk Decomposition Dashboard
====================================
Streamlit app that:
  1. Runs a joint (multivariate) regression of stock returns on a set of
     factor/ETF returns to estimate a loadings matrix B.                [S4.1]
  2. Combines B with current portfolio positions to compute dollar
     factor exposures.                                                  [S7.1]
  3. Decomposes total portfolio risk into FACTOR and IDIOSYNCRATIC
     components, and further into per-factor contributions, in a table
     styled after Table 7.1 of the PDF.                                 [App 11.1.3]

All math lives in risk_engine.py; this file is purely presentation.
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta

from risk_engine import estimate_factor_model, decompose_portfolio_risk, build_summary_rows
from data_sources import fetch_returns, YFINANCE_AVAILABLE, YFINANCE_IMPORT_ERROR

st.set_page_config(page_title="Factor Risk Decomposition", layout="wide")

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Demo data (used when no files are uploaded, so the app is testable offline)
# --------------------------------------------------------------------------- #

@st.cache_data
def make_demo_data(seed: int = 42, n_days: int = 500):
    rng = np.random.default_rng(seed)
    tickers = ["AAPL", "MSFT", "NVDA", "AMD", "XOM", "CVX", "JPM", "WMT", "GOOGL", "META"]
    factors = ["MKT", "VALUE", "MOMENTUM", "TECH", "SEMIS"]
    dates = pd.bdate_range("2022-01-01", periods=n_days)

    factor_vols = {"MKT": 0.010, "VALUE": 0.006, "MOMENTUM": 0.008, "TECH": 0.012, "SEMIS": 0.018}
    F = pd.DataFrame(index=dates)
    mkt = rng.normal(0, factor_vols["MKT"], n_days)
    F["MKT"] = mkt
    F["VALUE"] = rng.normal(0, factor_vols["VALUE"], n_days) - 0.15 * mkt
    F["MOMENTUM"] = rng.normal(0, factor_vols["MOMENTUM"], n_days)
    F["TECH"] = rng.normal(0, factor_vols["TECH"], n_days) + 0.5 * mkt
    F["SEMIS"] = 0.7 * F["TECH"] + rng.normal(0, 0.010, n_days)

    true_betas = {
        "AAPL":  {"MKT": 1.10, "VALUE": -0.20, "MOMENTUM": 0.30, "TECH": 0.80, "SEMIS": 0.10},
        "MSFT":  {"MKT": 1.00, "VALUE": -0.30, "MOMENTUM": 0.20, "TECH": 0.90, "SEMIS": 0.05},
        "GOOGL": {"MKT": 1.05, "VALUE": -0.25, "MOMENTUM": 0.15, "TECH": 0.85, "SEMIS": 0.05},
        "META":  {"MKT": 1.20, "VALUE": -0.35, "MOMENTUM": 0.45, "TECH": 0.75, "SEMIS": 0.10},
        "NVDA":  {"MKT": 1.30, "VALUE": -0.50, "MOMENTUM": 0.60, "TECH": 0.70, "SEMIS": 1.00},
        "AMD":   {"MKT": 1.40, "VALUE": -0.40, "MOMENTUM": 0.50, "TECH": 0.50, "SEMIS": 1.10},
        "XOM":   {"MKT": 0.90, "VALUE": 0.60, "MOMENTUM": -0.10, "TECH": -0.10, "SEMIS": 0.00},
        "CVX":   {"MKT": 0.85, "VALUE": 0.55, "MOMENTUM": -0.05, "TECH": -0.05, "SEMIS": 0.00},
        "JPM":   {"MKT": 1.05, "VALUE": 0.40, "MOMENTUM": 0.10, "TECH": 0.00, "SEMIS": 0.00},
        "WMT":   {"MKT": 0.50, "VALUE": 0.20, "MOMENTUM": 0.00, "TECH": 0.00, "SEMIS": 0.00},
    }

    R = pd.DataFrame(index=dates)
    for tkr in tickers:
        betas = true_betas[tkr]
        ret = sum(betas[f] * F[f] for f in factors)
        idio = rng.normal(0, 0.015, n_days)
        R[tkr] = ret + idio

    positions = pd.Series({
        "AAPL": 20.0, "MSFT": 15.0, "GOOGL": 10.0, "META": 8.0,
        "NVDA": 25.0, "AMD": 10.0,
        "XOM": -15.0, "CVX": -10.0, "JPM": 5.0, "WMT": -5.0,
    }, name="NMV")

    default_groups = {"MKT": "Market", "VALUE": "Style", "MOMENTUM": "Style",
                       "TECH": "Industry", "SEMIS": "Industry"}
    return R, F, positions, default_groups


# --------------------------------------------------------------------------- #
# Sidebar: data input
# --------------------------------------------------------------------------- #

st.sidebar.title("Data Input")
mode = st.sidebar.radio(
    "Data source",
    ["Use demo data", "Upload my own CSVs", "Fetch live data (yfinance)"],
    index=0,
)

if mode == "Upload my own CSVs":
    st.sidebar.markdown(
        "**Expected formats** (all wide-format CSVs, decimal daily returns e.g. `0.012` = 1.2%):\n\n"
        "- **Stock returns**: first column = date, remaining columns = one per ticker\n"
        "- **Factor/ETF returns**: first column = date, remaining columns = one per factor/ETF\n"
        "- **Positions**: two columns, `Ticker` and `NMV` (dollar net market value; "
        "use the same currency unit throughout, e.g. all in $M)"
    )
    stock_file = st.sidebar.file_uploader("Stock returns CSV", type=["csv"])
    factor_file = st.sidebar.file_uploader("Factor / ETF returns CSV", type=["csv"])
    positions_file = st.sidebar.file_uploader("Positions CSV (Ticker, NMV)", type=["csv"])

    if stock_file and factor_file and positions_file:
        stock_returns = pd.read_csv(stock_file, index_col=0, parse_dates=True)
        factor_returns = pd.read_csv(factor_file, index_col=0, parse_dates=True)
        pos_df = pd.read_csv(positions_file)
        pos_df.columns = [c.strip() for c in pos_df.columns]
        positions = pd.Series(pos_df["NMV"].values, index=pos_df["Ticker"].values, name="NMV")
        default_groups = {}
    else:
        st.info("Upload all three CSVs to proceed, or switch to demo data in the sidebar.")
        st.stop()

elif mode == "Fetch live data (yfinance)":
    if not YFINANCE_AVAILABLE:
        st.sidebar.error(f"yfinance is not installed ({YFINANCE_IMPORT_ERROR}). Run: pip install yfinance")
        st.stop()

    st.sidebar.caption(
        "Pulls real daily prices from Yahoo Finance via yfinance and converts "
        "them to returns. Requires outbound internet access — this will fail "
        "in network-locked sandboxes, but works locally or in a GitHub Codespace."
    )

    st.sidebar.markdown("**Positions**")
    st.sidebar.caption(
        "Enter each ticker and the number of shares you hold "
        "(negative shares = short position). The app fetches the latest "
        "price and computes dollar NMV = shares × price automatically — "
        "this list also determines which stocks get fetched, so there's no "
        "separate ticker box to keep in sync."
    )
    default_positions_df = pd.DataFrame({
        "Ticker": ["AAPL", "MSFT", "NVDA"],
        "Shares": [10, 10, 5],
    })
    positions_input_df = st.sidebar.data_editor(
        default_positions_df, num_rows="dynamic", key="positions_editor", width="stretch"
    )
    positions_input_df = positions_input_df.dropna(subset=["Ticker"])
    positions_input_df["Ticker"] = positions_input_df["Ticker"].astype(str).str.strip().str.upper()
    positions_input_df = positions_input_df[positions_input_df["Ticker"] != ""]
    stock_tickers = sorted(set(positions_input_df["Ticker"]))

    st.sidebar.markdown("**Factor / ETF mapping**")
    st.sidebar.caption("Edit tickers and give each a descriptive factor name.")
    default_factor_map = pd.DataFrame({
        "Ticker": ["SPY", "VTV", "MTUM", "XLK", "SOXX"],
        "Factor Name": ["Market", "Value", "Momentum", "Technology", "Semiconductors"],
    })
    factor_map_df = st.sidebar.data_editor(
        default_factor_map, num_rows="dynamic", key="factor_map_editor", width="stretch"
    )
    factor_map_df = factor_map_df.dropna(subset=["Ticker", "Factor Name"])
    factor_tickers = [t.strip().upper() for t in factor_map_df["Ticker"].tolist() if str(t).strip()]
    ticker_to_name = {
        str(t).strip().upper(): str(n).strip()
        for t, n in zip(factor_map_df["Ticker"], factor_map_df["Factor Name"])
    }

    col_a, col_b = st.sidebar.columns(2)
    end_date_default = date.today()
    start_date_default = end_date_default - timedelta(days=730)
    start_date = col_a.date_input("Start date", value=start_date_default)
    end_date = col_b.date_input("End date", value=end_date_default)

    fetch_clicked = st.sidebar.button("📡 Fetch / Refresh data", type="primary")

    if fetch_clicked:
        if not stock_tickers:
            st.sidebar.error("Add at least one ticker to the Positions table first.")
            st.stop()
        with st.spinner("Fetching prices from Yahoo Finance..."):
            try:
                stock_returns, stock_prices, failed_stocks = fetch_returns(stock_tickers, start_date, end_date)
            except Exception as e:
                st.sidebar.error(f"Stock fetch failed: {e}")
                st.stop()
            try:
                factor_returns_raw, _factor_prices, failed_factors = fetch_returns(
                    factor_tickers, start_date, end_date
                )
            except Exception as e:
                st.sidebar.error(f"Factor/ETF fetch failed: {e}")
                st.stop()
            rename_map = {t: ticker_to_name[t] for t in factor_returns_raw.columns if t in ticker_to_name}
            factor_returns = factor_returns_raw.rename(columns=rename_map)

            # NMV = shares * latest available closing price -- computed here
            # rather than asked of the user, since we already have the price
            # data from the same fetch used for the regression.
            latest_prices = stock_prices.iloc[-1]
            shares_map = dict(zip(positions_input_df["Ticker"], positions_input_df["Shares"]))
            nmv_dict, detail_rows, skipped_no_price = {}, [], []
            for tkr, shares in shares_map.items():
                if pd.isna(shares):
                    continue
                if tkr in latest_prices.index:
                    price = float(latest_prices[tkr])
                    nmv = float(shares) * price
                    nmv_dict[tkr] = nmv
                    detail_rows.append({"Ticker": tkr, "Shares": shares, "Latest Price": price, "NMV": nmv})
                else:
                    skipped_no_price.append(tkr)

            st.session_state["yf_stock_returns"] = stock_returns
            st.session_state["yf_factor_returns"] = factor_returns
            st.session_state["yf_failed"] = failed_stocks + failed_factors
            st.session_state["yf_nmv"] = nmv_dict
            st.session_state["yf_position_detail"] = pd.DataFrame(detail_rows).set_index("Ticker") \
                if detail_rows else pd.DataFrame(columns=["Shares", "Latest Price", "NMV"])
            st.session_state["yf_skipped_no_price"] = skipped_no_price

    if "yf_stock_returns" not in st.session_state:
        st.info("Enter your positions and date range in the sidebar, then click **Fetch / Refresh data**.")
        st.stop()

    stock_returns = st.session_state["yf_stock_returns"]
    factor_returns = st.session_state["yf_factor_returns"]
    if st.session_state.get("yf_failed"):
        st.warning("Some tickers could not be fetched: " + "; ".join(st.session_state["yf_failed"]))
    if st.session_state.get("yf_skipped_no_price"):
        st.warning(
            "No price available for: " + ", ".join(st.session_state["yf_skipped_no_price"])
            + " — excluded from NMV. Check the ticker symbol."
        )

    with st.expander("💵 Computed positions (shares × latest price)", expanded=False):
        st.dataframe(
            st.session_state["yf_position_detail"].style.format(
                {"Shares": "{:,.0f}", "Latest Price": "{:,.2f}", "NMV": "{:,.2f}"}
            ),
            width="stretch",
        )

    positions = pd.Series(st.session_state["yf_nmv"], name="NMV")
    positions = positions[positions != 0]
    if positions.empty:
        st.info("No non-zero positions yet — check your Shares entries in the sidebar.")
        st.stop()

    # Heuristic default grouping from factor labels, purely for display; user can override below.
    default_groups = {}
    for name in factor_returns.columns:
        lname = name.lower()
        if "market" in lname or lname in ("mkt", "spy"):
            default_groups[name] = "Market"
        elif "value" in lname or "momentum" in lname or "quality" in lname or "size" in lname:
            default_groups[name] = "Style"
        else:
            default_groups[name] = "Industry"

else:
    stock_returns, factor_returns, positions, default_groups = make_demo_data()
    st.sidebar.success("Using synthetic demo data (10 stocks, 5 factors, 500 trading days).")
    with st.sidebar.expander("Why synthetic data?"):
        st.write(
            "This mode simulates returns from **known** true betas — useful for "
            "confirming the regression recovers sensible exposures before you "
            "point the app at your real data (uploaded CSVs or live yfinance data)."
        )

st.sidebar.divider()
min_obs = st.sidebar.slider("Minimum overlapping observations required", 30, 250, 60, step=10)

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #

st.title("📊 Factor Risk Decomposition Dashboard")
st.caption(
    "Multivariate factor regression → portfolio exposures → factor / idiosyncratic "
    "risk decomposition, implemented per *Advanced Portfolio Management* (Paleologo, 2021)."
)

with st.expander("📐 Methodology & formula reference (click to expand)", expanded=False):
    st.markdown(r"""
**1. Multi-factor model** — each stock is regressed *jointly* on all factors at once
(not one ETF at a time), so each beta reflects exposure net of the others:

$$ r_{i,t} = \alpha_i + \beta_{i,1} f_{1,t} + \dots + \beta_{i,m} f_{m,t} + \epsilon_{i,t} $$

*(Ch. 4, Eq. 4.1)*. The residual vol per stock is annualized as
$\sigma_{\epsilon,\text{annual}} = \sigma_{\epsilon,\text{daily}} \times \sqrt{252}$ *(Ch. 4.2 FAQ)*.

**2. Portfolio factor exposure** — dollar exposure to factor $k$ is the
sum-product of loadings and position sizes:

$$ b = B' w \qquad \text{(Ch. 7, Eq. 7.1)} $$

**3. Risk decomposition** — factor and idiosyncratic variance are independent, so they
add (Pythagoras, Ch. 3.4.2 / Appendix 11.1.3):

$$ \text{Var(total)} = \underbrace{b' \,\Omega_f\, b}_{\text{factor variance}} \;+\; \underbrace{w' \Omega_\epsilon w}_{\text{idio variance}} $$

with $\Omega_\epsilon$ diagonal (idiosyncratic returns assumed uncorrelated).
The **% idio variance** is $p = \text{idio\_var} / \text{total\_var}$ *(Ch. 7.2.1)* — the
book recommends generally keeping this **above ~70–75%** for a book that's supposed to
be expressing idiosyncratic, not factor, views.

**4. Per-factor contribution to variance (%Var)** uses the exact Euler decomposition of a
homogeneous-degree-2 function, so contributions sum precisely to the factor total (this is
how the PDF's Table 7.1 apportions correlated style/industry risk rather than double-counting it):

$$ \%\text{Var}_k = \frac{b_k \,(\Omega_f b)_k}{\text{Var(total)}} $$

**5. Marginal Contribution to Factor Risk (MCFR)** — the \$ change in factor risk per \$1
change in a given exposure, holding others fixed *(Appendix 11.1.5)*:

$$ \text{MCFR}_k = \frac{(\Omega_f b)_k}{\sqrt{b'\Omega_f b}} $$

applied identically at the per-stock level using $B\Omega_f b$ instead of $\Omega_f b$.
""")

# --------------------------------------------------------------------------- #
# Run the factor model
# --------------------------------------------------------------------------- #

try:
    with st.spinner("Running joint multivariate factor regression..."):
        result = estimate_factor_model(stock_returns, factor_returns, min_obs=min_obs)
except ValueError as e:
    st.error(str(e))
    st.stop()

# Let the user tag factors into groups for the summary table (Market/Style/Industry/etc.)
st.sidebar.divider()
st.sidebar.subheader("Factor grouping (for display only)")
group_options = ["Market", "Style", "Industry", "Technical", "Custom", "Factor"]
factor_groups = {}
for f in result.factor_names:
    default = default_groups.get(f, "Factor")
    default_idx = group_options.index(default) if default in group_options else group_options.index("Factor")
    factor_groups[f] = st.sidebar.selectbox(f"'{f}'", group_options, index=default_idx, key=f"group_{f}")

# --------------------------------------------------------------------------- #
# Regression diagnostics
# --------------------------------------------------------------------------- #

st.header("1. Factor Model Estimation")
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Loadings matrix (B)")
    st.caption("Beta of each stock to each factor, net of the other factors (joint regression).")
    st.dataframe(result.loadings.style.format("{:.2f}").background_gradient(cmap="RdBu_r", vmin=-1.5, vmax=1.5),
                 width="stretch")

with col2:
    st.subheader("Regression fit")
    diag = pd.DataFrame({
        "R²": result.r_squared,
        "Idio Vol (ann. %)": result.idio_vol_annual * 100,
        "Obs": result.n_obs.astype(int),
    })
    st.dataframe(diag.style.format({"R²": "{:.2f}", "Idio Vol (ann. %)": "{:.1f}", "Obs": "{:.0f}"}),
                 width="stretch")
    low_r2 = diag[diag["R²"] < 0.10]
    if len(low_r2):
        st.warning(
            f"{len(low_r2)} stock(s) have R² < 10% ({', '.join(low_r2.index)}). "
            "Per the PDF's caution (Ch. 3.3), low explanatory power means most of the "
            "stock's variance is being classified as idiosyncratic almost by default — "
            "sensible, but double-check these aren't just noisy/short histories."
        )

with st.expander("Factor covariance matrix (Ω_f, annualized)"):
    st.dataframe(result.factor_cov_annual.style.format("{:.4f}"), width="stretch")

# --------------------------------------------------------------------------- #
# Portfolio risk decomposition
# --------------------------------------------------------------------------- #

st.header("2. Portfolio Risk Decomposition")

decomp = decompose_portfolio_risk(
    positions, result.loadings, result.idio_vol_annual, result.factor_cov_annual,
    factor_groups=factor_groups,
)

if decomp.missing_tickers:
    st.warning(
        f"No regression betas available for: {', '.join(decomp.missing_tickers)} "
        "(not in the stock-returns file). These positions were excluded from the "
        "decomposition below."
    )

gmv = positions.abs().sum()
nmv = positions.sum()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("GMV", f"{gmv:,.1f}")
k2.metric("NMV", f"{nmv:,.1f}")
k3.metric("Total Vol", f"{decomp.total_vol:,.2f}")
k4.metric("Factor Vol", f"{decomp.factor_vol:,.2f}", f"{decomp.pct_factor*100:.1f}% of var")
k5.metric("Idio Vol", f"{decomp.idio_vol:,.2f}", f"{decomp.pct_idio*100:.1f}% of var")

if decomp.pct_idio < 0.70:
    st.info(
        f"**% idio variance = {decomp.pct_idio*100:.1f}%**, below the PDF's suggested "
        "70% floor (rarely below 50%) for a portfolio meant to express stock-specific "
        "views rather than factor bets (Ch. 7.2.1)."
    )
else:
    st.success(f"**% idio variance = {decomp.pct_idio*100:.1f}%** — within the PDF's recommended range (≥70%).")

st.subheader("Full risk decomposition table")
st.caption("Styled after Table 7.1 in the PDF: summary rows (TOTAL/IDIO/FACTOR) followed by each factor's contribution.")

summary_rows = build_summary_rows(decomp)
summary_display = summary_rows.copy()
summary_display.insert(0, "Group", ["", "", ""])
summary_display.insert(2, "$Exp", [np.nan, np.nan, np.nan])
summary_display.insert(4, "MCFR", [np.nan, np.nan, np.nan])
summary_display = summary_display[["Group", "$Exp", "$Vol", "%Var", "MCFR"]]

factor_display = decomp.factor_table.rename(columns={"$Vol (standalone)": "$Vol"})[["Group", "$Exp", "$Vol", "%Var", "MCFR"]]

full_table = pd.concat([summary_display, factor_display])
st.dataframe(
    full_table.style.format({"$Exp": "{:,.2f}", "$Vol": "{:,.2f}", "%Var": "{:,.2f}", "MCFR": "{:,.3f}"}, na_rep="")
    .apply(lambda row: ["font-weight: bold; background-color: #f0f2f6" if row.name in
                         ["TOTAL", "IDIO", "FACTOR"] else "" for _ in row], axis=1),
    width="stretch",
)

csv_buf = io.StringIO()
full_table.to_csv(csv_buf)
st.download_button("⬇ Download decomposition table (CSV)", csv_buf.getvalue(),
                    file_name="risk_decomposition.csv", mime="text/csv")

# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #

st.subheader("Visualizations")
c1, c2 = st.columns(2)

with c1:
    pie_df = pd.DataFrame({
        "Component": ["Idiosyncratic", "Factor"],
        "Variance": [decomp.idio_var, decomp.factor_var],
    })
    fig_pie = px.pie(pie_df, names="Component", values="Variance",
                      title="Total Variance: Factor vs. Idiosyncratic",
                      color="Component",
                      color_discrete_map={"Idiosyncratic": "#2E86AB", "Factor": "#C73E1D"})
    st.plotly_chart(fig_pie, width="stretch")

with c2:
    bar_df = decomp.factor_table.reset_index().rename(columns={"index": "Factor"})
    fig_bar = px.bar(bar_df, x="Factor", y="%Var", color="Group",
                      title="% of Total Variance by Factor (Euler contribution)",
                      text_auto=".1f")
    fig_bar.add_hline(y=0, line_color="black", line_width=1)
    st.plotly_chart(fig_bar, width="stretch")

c3, c4 = st.columns(2)
with c3:
    fig_exp = px.bar(bar_df, x="Factor", y="$Exp", color="Group",
                      title="Dollar Factor Exposure (b = B'w)", text_auto=".1f")
    fig_exp.add_hline(y=0, line_color="black", line_width=1)
    st.plotly_chart(fig_exp, width="stretch")

with c4:
    fig_mcfr = px.bar(bar_df, x="Factor", y="MCFR", color="Group",
                       title="Marginal Contribution to Factor Risk (MCFR)", text_auto=".3f")
    fig_mcfr.add_hline(y=0, line_color="black", line_width=1)
    st.plotly_chart(fig_mcfr, width="stretch")
    st.caption("Reduction in **factor** $Vol per $1 cut in that factor's exposure, others held fixed (App. 11.1.5).")

# --------------------------------------------------------------------------- #
# Per-stock detail
# --------------------------------------------------------------------------- #

st.header("3. Per-Stock Detail")
st.caption(
    "Per-stock marginal contribution to **total factor risk** — i.e. how much factor "
    "risk changes per $1 change in that stock's position, holding all other positions "
    "fixed (Appendix 11.1.5, analogous to Table 7.2)."
)
stock_display = decomp.stock_table.copy()
st.dataframe(
    stock_display.style.format({
        "NMV": "{:,.2f}", "Idio Vol %": "{:.1f}", "Idio $Vol": "{:,.2f}", "MCFR (factor risk)": "{:,.3f}",
    }),
    width="stretch",
)

fig_stock = px.bar(stock_display.reset_index().rename(columns={"index": "Ticker"}),
                    x="Ticker", y="NMV", color="MCFR (factor risk)",
                    color_continuous_scale="RdBu_r", title="Position size, colored by MCFR to factor risk")
fig_stock.add_hline(y=0, line_color="black", line_width=1)
st.plotly_chart(fig_stock, width="stretch")

st.divider()
st.caption(
    "Built with a joint multivariate regression against user-supplied factor/ETF returns. "
    "See the methodology expander above for the full formula reference to "
    "*Advanced Portfolio Management* (Paleologo, 2021)."
)
