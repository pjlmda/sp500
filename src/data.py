import os
import pandas as pd
import yfinance as yf

import config

# Macro / cross-asset series fetched alongside S&P 500.
# Using ETFs where possible — more reliable on yfinance than index tickers.
MACRO_TICKERS = {
    "tnx":  "^TNX",   # 10-Year Treasury yield
    "irx":  "^IRX",   # 3-Month Treasury yield
    "dxy":  "UUP",    # Dollar strength (Invesco USD Bullish ETF)
    "gold": "GLD",    # Gold (SPDR Gold Shares)
    "xly":  "XLY",    # Consumer Discretionary (cyclical)
    "xlp":  "XLP",    # Consumer Staples (defensive)
    "hyg":  "HYG",    # High-Yield bonds (credit risk appetite)
    "tlt":  "TLT",    # Long-Term Treasuries (flight-to-safety)
}


def _download(ticker: str, start: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index.name = "date"
    return df.dropna()


def _fetch_macro(start: str) -> pd.DataFrame:
    frames = {}
    for col, ticker in MACRO_TICKERS.items():
        try:
            raw = _download(ticker, start)
            frames[col] = raw["Close"]
        except Exception as e:
            print(f"  Warning: could not fetch {ticker} ({col}): {e}")
    return pd.DataFrame(frames)


def load(force_refresh: bool = False) -> pd.DataFrame:
    """
    Returns OHLCV + VIX + macro DataFrame, cached to parquet.
    Columns: open, high, low, close, volume, vix,
             tnx, irx, dxy, gold, xly, xlp, hyg, tlt
    """
    cache = os.path.join(config.DATA_DIR, "sp500_combined.parquet")

    if not force_refresh and os.path.exists(cache):
        return pd.read_parquet(cache)

    os.makedirs(config.DATA_DIR, exist_ok=True)

    print("Fetching S&P 500...")
    sp = _download(config.TICKER, config.START_DATE)
    sp = sp[["Open", "High", "Low", "Close", "Volume"]]
    sp.columns = ["open", "high", "low", "close", "volume"]

    print("Fetching VIX...")
    vix = _download("^VIX", config.START_DATE)[["Close"]].rename(columns={"Close": "vix"})

    print("Fetching macro / cross-asset tickers...")
    macro = _fetch_macro(config.START_DATE)

    df = sp.join(vix, how="left").join(macro, how="left")

    # Guarantee every expected macro column exists — fill with NaN if ticker failed
    macro_cols = ["vix"] + list(MACRO_TICKERS.keys())
    for col in macro_cols:
        if col not in df.columns:
            print(f"  Warning: {col} missing from data, filling with NaN")
            df[col] = float("nan")

    # Forward-fill for non-trading day gaps, backward-fill for any leading NaN,
    # then zero-fill as a last resort so feature engineering never sees NaN
    df[macro_cols] = df[macro_cols].ffill().bfill().fillna(0)
    df = df.dropna(subset=["open", "close"])

    df.to_parquet(cache)
    return df
