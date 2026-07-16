"""
data_sources.py
================
Live market data via yfinance.

This automates the manual step the PDF describes in Ch. 3.1:
    "you can pull SYF and SP500 returns from a website or Bloomberg for
    the past year, estimate their vols, then perform a quick regression
    in Excel to estimate the beta of SYF to SP500..."

Tickers are fetched ONE AT A TIME (not as a single batched yf.download
call) so that:
  - one bad/delisted ticker doesn't take down the whole fetch
  - each ticker's result is cached independently, so re-running the app
    with a slightly different ticker list doesn't re-hit the network for
    tickers you already have
  - the price-column shape is unambiguous (yfinance's batched download
    returns different column layouts depending on version/args, which is
    a common source of subtle bugs)

NOTE: this module requires outbound internet access to
query1/query2.finance.yahoo.com. It will NOT work in network-sandboxed
environments (this was verified directly: such environments return
"Host not in allowlist" errors). It works fine in a normal local Python
environment, or in a GitHub Codespace, which have standard outbound
internet access.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
    YFINANCE_IMPORT_ERROR = None
except ImportError as e:
    YFINANCE_AVAILABLE = False
    YFINANCE_IMPORT_ERROR = str(e)


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_single_price_series(ticker: str, start: str, end: str) -> pd.Series:
    """
    Fetch a single ticker's adjusted-close daily price series.
    Cached for 1 hour (ttl=3600) so repeated app interactions (changing
    factor groupings, min_obs slider, etc.) don't re-hit the network.
    Raises ValueError with a clear message on failure.
    """
    hist = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
    if hist is None or hist.empty or "Close" not in hist.columns:
        raise ValueError(
            f"No price data returned for '{ticker}'. Check the ticker symbol, "
            f"the date range, and your internet connection."
        )
    s = hist["Close"].copy()
    # yfinance sometimes returns a tz-aware index; normalize to naive dates
    # so it aligns cleanly across tickers.
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s.name = ticker
    return s


def fetch_returns(tickers: list[str], start, end) -> tuple[pd.DataFrame, list[str]]:
    """
    Fetch adjusted-close prices for each ticker, align on common trading
    dates, and convert to simple (not log) daily returns -- matching the
    "decimal daily return" convention used throughout risk_engine.py.

    Parameters
    ----------
    tickers : list of ticker symbols (case-insensitive, whitespace-trimmed)
    start, end : anything pandas/yfinance accepts as a date (str or date)

    Returns
    -------
    (returns_df, failed_tickers)
        returns_df : DataFrame, index=dates, columns=tickers that succeeded
        failed_tickers : list of "TICKER (reason)" strings for any ticker
                         that could not be fetched
    """
    if not YFINANCE_AVAILABLE:
        raise ImportError(
            f"yfinance is not installed ({YFINANCE_IMPORT_ERROR}). "
            f"Run: pip install yfinance"
        )

    clean_tickers = [t.strip().upper() for t in tickers if t.strip()]
    if not clean_tickers:
        raise ValueError("No tickers were provided.")

    start_str, end_str = str(start), str(end)

    series_list, failed = [], []
    for t in clean_tickers:
        try:
            series_list.append(_fetch_single_price_series(t, start_str, end_str))
        except Exception as e:  # noqa: BLE001 -- surface any failure per-ticker, don't crash the batch
            failed.append(f"{t} ({e})")

    if not series_list:
        raise ValueError(
            "None of the requested tickers could be fetched. "
            + "; ".join(failed)
        )

    prices = pd.concat(series_list, axis=1, join="inner")
    if len(prices) < 2:
        raise ValueError(
            "Fewer than 2 overlapping trading days across the requested tickers "
            "-- widen the date range."
        )
    returns = prices.pct_change().dropna(how="all")
    return returns, failed
