"""Pull monthly stock returns from CRSP (via WRDS) for a short list of large,
well-known US stocks, along with the CRSP value-weighted market return and the
one-month Treasury bill rate.

You do NOT need to run this file to complete HW 0. It requires a WRDS account,
and yours may not be approved yet. The small extract that it produces is
already saved in `data_manual/`, and the tests and the notebook read that file.
The script is here so that you can see exactly where the data came from: in a
reproducible analytical pipeline, every number traces back to code.

Once your WRDS account is active, you can run it yourself:

    python ./src/pull_crsp.py

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

import pandas as pd
import wrds

import config

DATA_DIR = config.DATA_DIR
MANUAL_DATA_DIR = config.MANUAL_DATA_DIR
WRDS_USERNAME = config.WRDS_USERNAME

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


def load_extract(data_dir=MANUAL_DATA_DIR):
    path = data_dir / "crsp_monthly_returns.csv"
    return pd.read_csv(path, parse_dates=["date"], index_col="date")


if __name__ == "__main__":
    permnos_df = pull_permnos()
    returns_df = pull_monthly_returns(permnos_df["permno"])
    market_df = pull_market_and_riskfree()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    returns_df.to_parquet(DATA_DIR / "crsp_monthly_returns_long.parquet")

    extract = build_extract(permnos_df, returns_df, market_df)
    extract.to_csv(MANUAL_DATA_DIR / "crsp_monthly_returns.csv")
    print(permnos_df.to_string(index=False))
    print(extract.describe().T[["count", "mean", "std"]])
