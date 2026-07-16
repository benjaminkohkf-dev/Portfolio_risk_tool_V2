"""
risk_engine.py
==============
Core factor-risk decomposition engine.

Every formula below is implemented to match, section-for-section, the
math in:

    Paleologo, G. A. "Advanced Portfolio Management: A Quant's Guide
    for Fundamental Investors." Wiley, 2021.

Reference map (used in docstrings / comments as [Sx.y] or [App x.y]):

  [S3.4.2]  Single-stock risk = sqrt(market_vol^2 + idio_vol^2)          (Pythagoras)
  [S4.1]    Multi-factor return model: r = alpha + B f + eps            (Eq. 4.1)
  [S4.2]    Daily <-> annualized vol conversion: vol_annual = vol_daily * sqrt(252)
  [S7.1]    Portfolio factor exposure: b = B' w  (dollar exposure per factor) (Eq. 7.1)
  [S7.2.1]  %idio variance p = idio_var / total_var
  [App11.1.1] Asset covariance = B Omega_f B' + Omega_eps
  [App11.1.3] Portfolio variance = b' Omega_f b + w' Omega_eps w  (idio assumed diagonal)
  [App11.1.5] Marginal Contribution to Factor Risk (MCFR):
              MCFR_i = d/dw_i sqrt(w' B Omega_f B' w) = [B Omega_f b]_i / sqrt(b' Omega_f b)
  [T7.1]    Table 7.1-style decomposition: %Var, $Exp, $Vol, MCFR per factor/group

All "$" units below are whatever currency unit the user's position sizes
are denominated in (e.g. $M) -- the engine itself is unit-agnostic; it
simply assumes returns are in decimal (e.g. 0.01 = 1%) and positions are
in dollars (or $M, consistently).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

TRADING_DAYS = 252  # [S4.2] annualization constant used throughout the PDF


# --------------------------------------------------------------------------- #
# 1. Multi-factor model estimation  [S4.1 Eq 4.1] r = alpha + B f + eps
# --------------------------------------------------------------------------- #

@dataclass
class FactorModelResult:
    """Container for the estimated multi-factor model."""
    loadings: pd.DataFrame       # B: index=tickers, columns=factors           [S4.1]
    alpha_daily: pd.Series       # per-stock intercept (daily)                 [S4.1]
    idio_vol_annual: pd.Series   # per-stock annualized idiosyncratic vol      [S3.4.2/S4.2]
    r_squared: pd.Series         # per-stock regression fit quality
    n_obs: pd.Series             # observations used per stock
    residuals: pd.DataFrame      # daily idiosyncratic returns (for diagnostics)
    factor_cov_annual: pd.DataFrame  # Omega_f, annualized                    [App11.1.1]
    factor_names: list = field(default_factory=list)


def estimate_factor_model(
    stock_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    min_obs: int = 60,
) -> FactorModelResult:
    """
    Jointly (multivariate) regress each stock's returns on ALL factor
    returns simultaneously:

        r_i,t = alpha_i + beta_i1*f_1,t + ... + beta_im*f_m,t + eps_i,t   [S4.1 Eq 4.1]

    This is the "time-series approach" to factor-model estimation
    described in [S4.1]: factor returns are the observed ETF/index
    series; the betas (loadings) are estimated per stock.

    IMPORTANT: a joint (multivariate) regression is used -- not one
    univariate regression per ETF -- so that each beta represents the
    stock's exposure to that factor *net of* the other factors. This
    avoids the "impurity" problem the PDF flags in [S5.1.2]: a sector
    ETF or style ETF is not a pure factor and can bleed into other
    factors' betas if regressed one at a time.

    Parameters
    ----------
    stock_returns : DataFrame, index=dates, columns=tickers (decimal daily returns)
    factor_returns : DataFrame, index=dates, columns=factor names (decimal daily returns)
    min_obs : minimum overlapping observations required per stock

    Returns
    -------
    FactorModelResult
    """
    # Align on common dates and drop rows with any missing data, so that
    # every stock/factor pair is estimated on the identical sample.
    common = stock_returns.join(factor_returns, how="inner", lsuffix="_s", rsuffix="_f")
    common = common.dropna(how="any")

    factor_names = list(factor_returns.columns)
    tickers = list(stock_returns.columns)

    dates = common.index
    F = common[factor_names].to_numpy(dtype=float)              # T x m
    X = np.column_stack([np.ones(len(dates)), F])                # T x (m+1), intercept + factors

    n_obs = len(dates)
    if n_obs < min_obs:
        raise ValueError(
            f"Only {n_obs} overlapping observations between stock and factor "
            f"returns; need at least {min_obs}. Check date alignment."
        )

    m = len(factor_names)
    n = len(tickers)

    B = pd.DataFrame(index=tickers, columns=factor_names, dtype=float)
    alpha = pd.Series(index=tickers, dtype=float)
    idio_vol_annual = pd.Series(index=tickers, dtype=float)
    r_squared = pd.Series(index=tickers, dtype=float)
    n_obs_used = pd.Series(index=tickers, dtype=float)
    resid_df = pd.DataFrame(index=dates, columns=tickers, dtype=float)

    XtX_inv = np.linalg.pinv(X.T @ X)  # reused across stocks (same X for all)

    for tkr in tickers:
        y = common[tkr].to_numpy(dtype=float)
        coef = XtX_inv @ (X.T @ y)          # OLS coefficients: [intercept, beta_1..beta_m]
        y_hat = X @ coef
        resid = y - y_hat

        alpha[tkr] = coef[0]
        B.loc[tkr, :] = coef[1:]
        resid_df[tkr] = resid

        # idiosyncratic (residual) variance, annualized  [S4.2]: * 252
        dof = max(n_obs - (m + 1), 1)
        idio_var_daily = float((resid @ resid) / dof)
        idio_vol_annual[tkr] = np.sqrt(idio_var_daily * TRADING_DAYS)

        sst = float(np.sum((y - y.mean()) ** 2))
        ssr = float(resid @ resid)
        r_squared[tkr] = 1 - ssr / sst if sst > 0 else np.nan
        n_obs_used[tkr] = n_obs

    # Factor covariance matrix Omega_f, annualized  [App11.1.1] Omega_f,t
    factor_cov_daily = common[factor_names].cov()
    factor_cov_annual = factor_cov_daily * TRADING_DAYS

    return FactorModelResult(
        loadings=B,
        alpha_daily=alpha,
        idio_vol_annual=idio_vol_annual,
        r_squared=r_squared,
        n_obs=n_obs_used,
        residuals=resid_df,
        factor_cov_annual=factor_cov_annual,
        factor_names=factor_names,
    )


# --------------------------------------------------------------------------- #
# 2. Portfolio-level risk decomposition  [S7.1], [S7.2.1], [App 11.1.3/11.1.5]
# --------------------------------------------------------------------------- #

@dataclass
class RiskDecomposition:
    total_var: float
    factor_var: float
    idio_var: float
    total_vol: float
    factor_vol: float
    idio_vol: float
    pct_idio: float           # [S7.2.1] p = idio_var / total_var
    pct_factor: float
    exposures: pd.Series       # b = B' w                              [S7.1 Eq 7.1]
    factor_table: pd.DataFrame  # per-factor $Exp, $Vol, %Var, MCFR     [Table 7.1]
    stock_table: pd.DataFrame   # per-stock NMV, idio vol$, MCFR        [Table 7.2-style]
    missing_tickers: list = field(default_factory=list)


def decompose_portfolio_risk(
    weights: pd.Series,
    loadings: pd.DataFrame,
    idio_vol_annual: pd.Series,
    factor_cov_annual: pd.DataFrame,
    factor_groups: Optional[dict] = None,
) -> RiskDecomposition:
    """
    Decompose portfolio risk into factor and idiosyncratic components,
    following the risk-model machinery of Chapters 3-4 and 7 and
    Appendix 11.1.

    weights : Series of dollar Net Market Value (NMV) per ticker
    loadings : B, DataFrame (tickers x factors)                        [S4.1]
    idio_vol_annual : per-stock annualized idio vol (%, e.g. 0.30=30%) if
                       expressed as a *percentage* of NMV, OR in dollar
                       terms if `idio_vol_annual` already carries $ units.
                       Here we treat it as a PERCENTAGE (matches the PDF's
                       Table 3.4 "Daily Idio Vol (%)" convention) and
                       convert to dollars internally.
    factor_cov_annual : Omega_f, annualized factor covariance matrix
    factor_groups : optional {factor_name: group_name} mapping, purely
                    for display grouping (e.g. "Style"/"Industry"/"Market")
    """
    tickers = [t for t in weights.index if t in loadings.index]
    missing = set(weights.index) - set(loadings.index)
    w = weights.loc[tickers].astype(float)
    B = loadings.loc[tickers].astype(float)
    idio_pct = idio_vol_annual.loc[tickers].astype(float)

    factor_names = list(B.columns)
    Omega_f = factor_cov_annual.loc[factor_names, factor_names].to_numpy()

    # ---- Factor exposures: b = B' w  [S7.1 Eq 7.1] ----
    b = B.to_numpy().T @ w.to_numpy()                 # m-vector, dollar exposure per factor
    b = pd.Series(b, index=factor_names)

    # ---- Factor variance: b' Omega_f b  [App 11.1.3] ----
    Omega_f_b = Omega_f @ b.to_numpy()                 # m-vector
    factor_var = float(b.to_numpy() @ Omega_f_b)
    factor_var = max(factor_var, 0.0)
    factor_vol = np.sqrt(factor_var)

    # ---- Idiosyncratic variance: sum_i (w_i * idio_pct_i)^2  ----
    # [S3.4.2 Pythagoras] / [App 11.1.3] with diagonal Omega_eps.
    # idio_pct is a PERCENTAGE, so dollar idio vol per stock = w_i * idio_pct_i
    idio_dollar_vol = (w * idio_pct)
    idio_var = float((idio_dollar_vol ** 2).sum())
    idio_vol = np.sqrt(idio_var)

    # ---- Total variance: factor + idio (independent)  [App 11.1.3] ----
    total_var = factor_var + idio_var
    total_vol = np.sqrt(total_var)

    pct_idio = idio_var / total_var if total_var > 0 else np.nan
    pct_factor = factor_var / total_var if total_var > 0 else np.nan

    # ---- Per-factor table: %Var (Euler contribution), $Exp, $Vol, MCFR ----
    # %Var_k uses the Euler/homogeneous-degree-2 decomposition of variance:
    #   sum_k b_k * (Omega_f b)_k = b' Omega_f b = factor_var   (exact)
    # This automatically splits correlation-driven variance across the
    # correlated factors, consistent with the PDF's footnote in [T7.1]
    # that correlated style/industry contributions are apportioned rather
    # than double counted.
    with np.errstate(invalid="ignore", divide="ignore"):
        pct_var_factor_level = (b.to_numpy() * Omega_f_b) / total_var if total_var > 0 else np.zeros_like(b)
        # standalone (own-vol) dollar volatility of each factor bet, ignoring
        # cross-factor correlation -- matches the PDF's illustrative usage
        # ("cutting 12-Mo Momentum exposure by 50% halves its $Vol") [S7.1]
        own_vol = np.sqrt(np.diag(Omega_f))
        dollar_vol_standalone = np.abs(b.to_numpy()) * own_vol
        # MCFR at the factor level: marginal $ change in *factor* risk per
        # $1 change in that factor's exposure, holding others fixed.
        # Same functional form as [App 11.1.5], applied to factor risk
        # sqrt(b'Omega_f b) instead of stock-level portfolio risk.
        mcfr_factor = Omega_f_b / factor_vol if factor_vol > 0 else np.zeros_like(b)

    factor_table = pd.DataFrame({
        "$Exp": b.values,
        "$Vol (standalone)": dollar_vol_standalone,
        "%Var": pct_var_factor_level * 100,
        "MCFR": mcfr_factor,
    }, index=factor_names)

    if factor_groups:
        factor_table.insert(0, "Group", [factor_groups.get(f, "Factor") for f in factor_names])
    else:
        factor_table.insert(0, "Group", "Factor")

    factor_table = factor_table.sort_values("%Var", ascending=False)

    # ---- Per-stock table: NMV, idio $vol, stock-level MCFR to total factor risk ----
    # Stock-level MCFR follows [App 11.1.5] literally:
    #   MCFR_i = [B Omega_f b]_i / sqrt(b' Omega_f b)
    B_Omega_f_b = B.to_numpy() @ Omega_f_b            # n-vector
    with np.errstate(invalid="ignore", divide="ignore"):
        stock_mcfr = B_Omega_f_b / factor_vol if factor_vol > 0 else np.zeros(len(tickers))

    stock_table = pd.DataFrame({
        "NMV": w.values,
        "Idio Vol %": idio_pct.values * 100,
        "Idio $Vol": idio_dollar_vol.values,
        "MCFR (factor risk)": stock_mcfr,
    }, index=tickers)
    stock_table = stock_table.reindex(stock_table["NMV"].abs().sort_values(ascending=False).index)

    return RiskDecomposition(
        total_var=total_var,
        factor_var=factor_var,
        idio_var=idio_var,
        total_vol=total_vol,
        factor_vol=factor_vol,
        idio_vol=idio_vol,
        pct_idio=pct_idio,
        pct_factor=pct_factor,
        exposures=b,
        factor_table=factor_table,
        stock_table=stock_table,
        missing_tickers=sorted(missing),
    )


# --------------------------------------------------------------------------- #
# 3. Convenience: build a Table-7.1-style summary block (TOTAL/IDIO/FACTOR rows)
# --------------------------------------------------------------------------- #

def build_summary_rows(decomp: RiskDecomposition) -> pd.DataFrame:
    """
    Build the top summary rows (TOTAL / IDIO / FACTOR) exactly as they
    appear at the top of Table 7.1 in the PDF, in the same currency units
    as the input NMVs.
    """
    rows = pd.DataFrame(
        {
            "$Vol": [decomp.total_vol, decomp.idio_vol, decomp.factor_vol],
            "%Var": [100.0, decomp.pct_idio * 100, decomp.pct_factor * 100],
        },
        index=["TOTAL", "IDIO", "FACTOR"],
    )
    return rows
