# Factor Risk Decomposition Dashboard

A Streamlit app that takes a portfolio + a set of factor/ETF return series,
runs a **joint (multivariate) regression** to estimate each stock's exposure
to every factor simultaneously, and produces a full **factor vs.
idiosyncratic risk decomposition** — styled after Table 7.1 in
*Advanced Portfolio Management: A Quant's Guide for Fundamental Investors*
(Paleologo, 2021).

Three data sources are supported:
1. **Demo data** — synthetic, generated from known true betas (works instantly, no setup)
2. **Upload your own CSVs** — bring your own returns/positions
3. **Live data via yfinance** — pulls real daily prices from Yahoo Finance

## Why "joint" regression matters

If you regress a stock against one ETF at a time (e.g. momentum ETF, then
separately a tech ETF), each beta can "leak" into the others — a momentum
ETF that happens to be overweight tech will make your momentum beta partly
a tech bet. This app regresses each stock on **all factors at once**, so
each beta is net of the others. See the in-app "Methodology" expander for
the full formula reference.

---

## Quickstart (local machine)

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens with **synthetic demo data** already loaded, so you can see
the full dashboard immediately with zero setup.

---

## Option A: Upload your own CSVs

Switch "Data source" to **Upload my own CSVs** in the sidebar. You need
three files (see `sample_data/` for exact-format examples):

1. **Stock returns** — wide CSV, first column = date, one column per
   ticker, decimal daily returns (e.g. `0.012` = 1.2%).
2. **Factor / ETF returns** — same format, one column per factor/ETF.
3. **Positions** — two columns: `Ticker`, `NMV` (dollar net market value
   of your current holding in each name; use one consistent currency unit
   throughout, e.g. everything in $M).

Returns should be **decimal simple returns**, not log returns and not raw
prices — convert prices to returns (`prices.pct_change()`) before exporting.

---

## Option B: Live data via yfinance

Switch "Data source" to **Fetch live data (yfinance)**. In the sidebar:

1. Edit the **Positions** table directly — enter each **Ticker** and the
   **number of Shares** you hold (negative shares = short). This table is
   also what determines which stocks get fetched — there's no separate
   ticker list to keep in sync. Use the **"+"** button at the bottom of
   the table to add rows, or the row menu to delete one.
2. Edit the **factor/ETF mapping** table — each row is one ETF ticker and
   the factor name you want it to represent. Sensible starting points:

   | Ticker | Represents |
   |---|---|
   | `SPY` | Market |
   | `VTV` or `IWD` | Value |
   | `MTUM` | Momentum |
   | `XLK` | Technology (sector) |
   | `SOXX` or `SMH` | Semiconductors (industry) |
   | `QUAL` | Quality |
   | `USMV` | Low volatility |

   Pick ETFs relevant to *your* book — these are just a reasonable default set.
3. Set the **date range** (defaults to the last 2 years).
4. Click **Fetch / Refresh data**.

The app fetches the latest close price for each ticker and computes
**dollar NMV = Shares × Latest Price automatically** — you never enter a
dollar amount by hand. You can inspect the computed prices/NMVs in the
**"Computed positions (shares × latest price)"** expander once data has
loaded. If you add a new ticker to the Positions table after already
fetching, it'll be flagged as missing a price until you click Fetch again.

Data is cached for 1 hour per ticker/date-range combination, so tweaking
other settings (factor groupings, min-observations slider, or editing
share counts) afterward won't re-hit the network — NMV recalculates
instantly off the cached prices. Click the fetch button again any time
you want fresh prices.

**Important — this requires real outbound internet access.** It will
**not** work in network-locked sandboxes (this was directly verified: such
environments return `Host not in allowlist: query1/query2.finance.yahoo.com`
errors). It works fine on a normal local machine or in a GitHub Codespace
(see below), both of which have standard internet access.

---

## Running this in a new GitHub Codespace

This lets you run the dashboard entirely in the browser, with no local
Python install, and with full internet access for the yfinance mode.

### 1. Create a new repository and add these files

- Go to github.com/new, create a new repository (public or private,
  doesn't matter), leave it empty (no README/license needed).
- On the repo page, click **Add file -> Upload files**, and drag in all of
  these:
  - `app.py`
  - `risk_engine.py`
  - `data_sources.py`
  - `requirements.txt`
  - `README.md`
  - `.devcontainer/devcontainer.json` (keep the folder structure -- GitHub's
    upload UI preserves subfolders if you drag the whole `.devcontainer`
    folder, or use git locally if you prefer)
  - `.streamlit/config.toml`
  - `sample_data/stock_returns.csv`, `sample_data/factor_returns.csv`,
    `sample_data/positions.csv`
- Commit directly to `main`.

  (If you'd rather use git locally: `git init`, add these files, `git add -A`,
  `git commit -m "initial commit"`, then `git remote add origin <your-repo-url>`
  and `git push -u origin main`.)

### 2. Launch a Codespace

- On the repo's main page, click the green **Code** button.
- Select the **Codespaces** tab.
- Click **Create codespace on main**.
- Wait ~1-2 minutes while it builds the container. Because of the
  `.devcontainer/devcontainer.json` included here, it will **automatically
  run `pip install -r requirements.txt` for you** -- no manual setup needed.

### 3. Run the app

Once the Codespace opens (a VS Code environment in your browser), open a
terminal (**Terminal -> New Terminal**, or it may already be open) and run:

```bash
streamlit run app.py
```

A notification should pop up: "Your application running on port 8501 is
available." Click **Open in Browser**. (If you miss the popup, go to the
**Ports** tab at the bottom panel, find port `8501`, and click the globe
icon to open it.)

### 4. Use it

In the sidebar, select **Fetch live data (yfinance)**, enter your tickers
and factor ETFs, set a date range, click **Fetch / Refresh data** -- this
will work in Codespaces since it has normal internet access.

### Notes on Codespaces

- Free GitHub accounts get a monthly quota of Codespaces usage hours;
  check your usage at github.com/settings/billing.
- Stop the Codespace when you're done (**Codespaces** menu -> **Stop
  codespace**) to avoid using up your quota while idle.
- If you want to share the running app's URL with someone else, set the
  port's visibility to **Public** in the Ports tab (right-click the port
  row -> **Port Visibility -> Public**). By default forwarded ports are
  private to your GitHub account.

---

## What the dashboard shows

- **Loadings matrix (B)** -- each stock's beta to each factor, net of the
  others, with a regression-quality panel (R-squared, annualized idio
  vol, observation count).
- **Full risk decomposition table** -- TOTAL / IDIO / FACTOR summary rows,
  followed by a per-factor breakdown of dollar exposure, standalone dollar
  volatility, % contribution to total variance, and Marginal Contribution
  to Factor Risk (MCFR) -- the same structure as PDF Table 7.1.
- **% idiosyncratic variance** flagged against the PDF's guidance
  (generally keep it >=70%, rarely below 50%, Ch. 7.2.1).
- **Charts** -- factor vs. idio variance split, %Var by factor, dollar
  exposure by factor, MCFR by factor.
- **Per-stock table** -- position size, idio vol, and each stock's
  marginal contribution to total factor risk (Table 7.2-style).
- CSV download of the full decomposition table.

## Formula reference (all implemented in `risk_engine.py`)

| Concept | Formula | PDF reference |
|---|---|---|
| Multi-factor model | `r = alpha + B f + eps` | Ch. 4, Eq. 4.1 |
| Daily -> annualized vol | `vol_annual = vol_daily * sqrt(252)` | Ch. 4.2 FAQ |
| Portfolio factor exposure | `b = B' w` | Ch. 7, Eq. 7.1 |
| Portfolio variance | `total_var = b'Omega_f b + w'Omega_eps w` | Ch. 3.4.2 (Pythagoras) / App. 11.1.3 |
| % idio variance | `p = idio_var / total_var` | Ch. 7.2.1 |
| Per-factor %Var (Euler contribution) | `%Var_k = b_k (Omega_f b)_k / total_var` | consistent with Table 7.1's correlation-splitting footnote |
| Marginal Contribution to Factor Risk | `MCFR_k = (Omega_f b)_k / sqrt(b'Omega_f b)` | App. 11.1.5 |
| Per-stock MCFR (to total factor risk) | `MCFR_i = [B Omega_f b]_i / sqrt(b'Omega_f b)` | App. 11.1.5 |

## Files

- `app.py` -- Streamlit UI (presentation only)
- `risk_engine.py` -- all math, validated against synthetic data with
  known true betas (regression correctly recovers them)
- `data_sources.py` -- yfinance data fetching, with per-ticker error
  isolation and caching
- `requirements.txt`
- `.devcontainer/devcontainer.json` -- auto-configures a GitHub Codespace
- `.streamlit/config.toml` -- server settings for clean port forwarding
- `sample_data/` -- example CSVs in the exact upload format

## Notes / limitations

- This is the **time-series** factor-model approach (regress on known
  factor return series), one of three approaches the PDF describes
  (Ch. 4.1) -- the others being the **fundamental/characteristic**
  approach (loadings are known stock characteristics, factor returns are
  estimated) and the **statistical** approach (both estimated from
  returns alone, via PCA-like methods). If your ETFs are themselves
  correlated with each other (e.g. a "tech" and a "semiconductor" ETF),
  some correlation will still show up in your betas even with joint
  regression -- check the factor covariance matrix (in the "Factor
  covariance matrix" expander) to see how correlated your chosen
  factors are.
- Idiosyncratic returns are assumed uncorrelated across stocks (diagonal
  Omega_eps), matching the PDF's standard simplifying assumption.
- The yfinance mode fetches **adjusted close prices** and converts to
  simple daily returns; it needs real outbound internet access and will
  not work in network-restricted sandboxes.
- yfinance is an unofficial, community-maintained wrapper around Yahoo
  Finance's public endpoints -- it can occasionally break or rate-limit
  if Yahoo changes something on their end. If fetches start failing,
  check for a `pip install --upgrade yfinance`.
