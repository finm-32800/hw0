"""Get monthly stock returns from CRSP for a short list of large, well-known US
stocks, along with the CRSP value-weighted market return and the one-month
Treasury bill rate.

There are two ways to get the data, and both save the same file,
`_data/crsp_monthly_returns.csv`:

 - By default, download a cached copy of the extract. This needs no account,
   so it works before your WRDS access is approved.
 - With the environment variable `NO_CACHE=True`, pull the data fresh from
   WRDS. This needs a WRDS account (put your `WRDS_USERNAME` in `.env`).

Run it directly, or let `doit` run it for you:

    python ./src/pull_crsp.py
    NO_CACHE=True python ./src/pull_crsp.py

Either way, read the WRDS queries below. In a reproducible analytical
pipeline, every number traces back to code, and this is the code that produced
the cached copy.

Notes on the CRSP tables used here (CRSP "CIZ" format, Flat File Format 2.0):
 - crspm.msf_v2: monthly stock file. `mthret` is the total monthly return,
   including dividends and delisting returns.
 - crspm.stksecurityinfohist: security names and tickers over time. A ticker
   can change or be reused, so CRSP identifies a security by its PERMNO. We look
   up the PERMNO that holds each ticker as of END_DATE.
 - crsp_a_indexes.msix: CRSP market indices. `vwretd` is the value-weighted
   market return, including dividends.
 - ff.factors_monthly: Fama-French factors. `rf` is the one-month T-bill rate.
"""

import urllib.request

import pandas as pd

import config

DATA_DIR = config.DATA_DIR
WRDS_USERNAME = config.WRDS_USERNAME
NO_CACHE = config.NO_CACHE

EXTRACT_FILENAME = "crsp_monthly_returns.csv"
CACHED_EXTRACT_URL = (
    "https://drive.google.com/uc?export=download&id=149oTtKeAlHi_B41VLtlfBvFlZQUjdgnE"
)

START_DATE = "2000-01-01"
END_DATE = "2024-12-31"

TICKERS = [
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "AMZN",  # Amazon
    "NVDA",  # Nvidia
    "JPM",  # JPMorgan Chase
    "XOM",  # Exxon Mobil
    "JNJ",  # Johnson & Johnson
    "PG",  # Procter & Gamble
    "KO",  # Coca-Cola
    "WMT",  # Walmart
    "CAT",  # Caterpillar
    "BA",  # Boeing
]


def pull_permnos(tickers=TICKERS, as_of=END_DATE, wrds_username=WRDS_USERNAME):
    """Find the PERMNO of the common stock trading under each ticker on `as_of`."""
    ticker_list = ", ".join(f"'{t}'" for t in tickers)
    query = f"""
        SELECT permno, ticker, issuernm
        FROM crspm.stksecurityinfohist
        WHERE ticker IN ({ticker_list})
            AND secinfostartdt <= '{as_of}'
            AND '{as_of}' <= secinfoenddt
            AND sharetype = 'NS'
            AND securitytype = 'EQTY'
            AND securitysubtype = 'COM'
            AND usincflg = 'Y'
            AND issuertype IN ('ACOR', 'CORP')
    """
    import wrds  # imported here so that the cached download needs no WRDS setup

    db = wrds.Connection(wrds_username=wrds_username)
    df = db.raw_sql(query)
    db.close()
    return df


def pull_monthly_returns(
    permnos, start_date=START_DATE, end_date=END_DATE, wrds_username=WRDS_USERNAME
):
    """Pull total monthly returns for the given PERMNOs."""
    permno_list = ", ".join(str(int(p)) for p in permnos)
    query = f"""
        SELECT permno, mthcaldt, mthret
        FROM crspm.msf_v2
        WHERE permno IN ({permno_list})
            AND mthcaldt BETWEEN '{start_date}' AND '{end_date}'
    """
    import wrds

    db = wrds.Connection(wrds_username=wrds_username)
    df = db.raw_sql(query, date_cols=["mthcaldt"])
    db.close()
    return df


def pull_market_and_riskfree(
    start_date=START_DATE, end_date=END_DATE, wrds_username=WRDS_USERNAME
):
    """Pull the CRSP value-weighted market return and the one-month T-bill rate."""
    query_mkt = f"""
        SELECT caldt, vwretd
        FROM crsp_a_indexes.msix
        WHERE caldt BETWEEN '{start_date}' AND '{end_date}'
    """
    query_rf = f"""
        SELECT date, rf
        FROM ff.factors_monthly
        WHERE date BETWEEN '{start_date}' AND '{end_date}'
    """
    import wrds

    db = wrds.Connection(wrds_username=wrds_username)
    mkt = db.raw_sql(query_mkt, date_cols=["caldt"])
    rf = db.raw_sql(query_rf, date_cols=["date"])
    db.close()

    # The two tables date their rows differently (last trading day vs. first
    # day of the month), so align both on the calendar month.
    mkt["month"] = mkt["caldt"].dt.to_period("M")
    rf["month"] = rf["date"].dt.to_period("M")
    df = mkt.merge(rf, on="month", how="left")
    return df.set_index("month")[["vwretd", "rf"]]


def build_extract(permnos_df, returns_df, market_df):
    """Reshape to one row per month and one column per series."""
    returns_df = returns_df.merge(permnos_df[["permno", "ticker"]], on="permno")
    returns_df["month"] = returns_df["mthcaldt"].dt.to_period("M")
    wide = returns_df.pivot(index="month", columns="ticker", values="mthret")
    wide = wide[[t for t in TICKERS if t in wide.columns]]
    extract = wide.join(market_df.rename(columns={"vwretd": "MKT", "rf": "RF"}))
    extract.index = extract.index.to_timestamp(how="end").normalize()
    extract.index.name = "date"
    return extract.astype(float).round(6)


def pull_extract_from_wrds():
    """Build the extract from scratch with three WRDS queries."""
    permnos_df = pull_permnos()
    returns_df = pull_monthly_returns(permnos_df["permno"])
    market_df = pull_market_and_riskfree()
    return build_extract(permnos_df, returns_df, market_df)


def download_cached_extract(url=CACHED_EXTRACT_URL):
    """Download the cached copy of the extract."""
    with urllib.request.urlopen(url) as response:
        return pd.read_csv(response, parse_dates=["date"], index_col="date")


def load_extract(data_dir=DATA_DIR):
    path = data_dir / EXTRACT_FILENAME
    return pd.read_csv(path, parse_dates=["date"], index_col="date")


if __name__ == "__main__":
    if NO_CACHE:
        print("NO_CACHE=True: pulling from WRDS.")
        extract = pull_extract_from_wrds()
    else:
        print("Downloading the cached extract. Set NO_CACHE=True to pull from WRDS.")
        extract = download_cached_extract()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    extract.to_csv(DATA_DIR / EXTRACT_FILENAME)
    print(extract.describe().T[["count", "mean", "std"]])
