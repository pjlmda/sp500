import pandas as pd
import numpy as np

import config


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = (ema_fast - ema_slow) / series
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger(series: pd.Series, window: int = 20, n_std: float = 2.0):
    mid = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    pct_b = (series - lower) / (upper - lower)
    width = (upper - lower) / mid
    return pct_b, width


def build(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Build features from OHLCV+VIX DataFrame.

    Returns (features_df, feature_cols). The last row always has valid
    features but NaN targets — kept intentionally for next-day inference.
    """
    f = pd.DataFrame(index=df.index)
    daily_return = df["close"].pct_change()

    # --- Short-term lagged returns ---
    for lag in config.FEATURE_LAGS:
        f[f"return_lag_{lag}"] = daily_return.shift(lag)

    # --- Longer-term momentum ---
    for lag in [21, 63, 126, 252]:
        f[f"return_lag_{lag}"] = df["close"].pct_change(lag)

    # Classic momentum factor: 12-month return minus 1-month (skip last month)
    f["momentum_12_1"] = df["close"].pct_change(252) - df["close"].pct_change(21)

    # --- MA ratios ---
    for w in config.MA_WINDOWS:
        ma = df["close"].rolling(w).mean()
        f[f"ma_ratio_{w}"] = df["close"] / ma - 1

    # --- Volatility ---
    for w in config.VOLATILITY_WINDOWS:
        f[f"volatility_{w}"] = daily_return.rolling(w).std()

    # --- RSI ---
    f["rsi"] = _rsi(df["close"], config.RSI_WINDOW)

    # --- MACD ---
    f["macd"], f["macd_signal"], f["macd_hist"] = _macd(df["close"])

    # --- Bollinger Bands ---
    f["bb_pct_b"], f["bb_width"] = _bollinger(df["close"])

    # --- Volume ---
    for w in config.VOLUME_WINDOWS:
        f[f"volume_ratio_{w}"] = df["volume"] / df["volume"].rolling(w).mean()

    # --- Price action ---
    f["hl_range"] = (df["high"] - df["low"]) / df["close"]
    f["gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)

    # --- Calendar ---
    f["day_of_week"] = df.index.dayofweek
    f["month"] = df.index.month

    # --- VIX ---
    if "vix" in df.columns:
        f["vix_level"] = df["vix"]
        f["vix_change_1d"] = df["vix"].pct_change()
        f["vix_change_5d"] = df["vix"].pct_change(5)
        f["vix_ma_ratio_10"] = df["vix"] / df["vix"].rolling(10).mean() - 1
        f["vix_zscore_20"] = (
            (df["vix"] - df["vix"].rolling(20).mean()) / df["vix"].rolling(20).std()
        )

    # --- Yield curve (10Y - 3M) ---
    if "tnx" in df.columns and "irx" in df.columns:
        yc = df["tnx"] - df["irx"]
        f["yield_curve"] = yc
        f["yield_curve_change_5d"] = yc.diff(5)
        f["yield_curve_change_21d"] = yc.diff(21)
        f["yield_curve_zscore_63"] = (yc - yc.rolling(63).mean()) / yc.rolling(63).std()

    # --- Dollar strength ---
    if "dxy" in df.columns:
        f["dollar_change_1d"] = df["dxy"].pct_change()
        f["dollar_change_5d"] = df["dxy"].pct_change(5)
        f["dollar_ma_ratio_20"] = df["dxy"] / df["dxy"].rolling(20).mean() - 1

    # --- Gold (safe-haven flow) ---
    if "gold" in df.columns:
        f["gold_change_1d"] = df["gold"].pct_change()
        f["gold_change_5d"] = df["gold"].pct_change(5)
        f["gold_ma_ratio_20"] = df["gold"] / df["gold"].rolling(20).mean() - 1
        # Gold relative to S&P500 (rising ratio = risk-off rotation)
        f["gold_spx_ratio_change_21d"] = (df["gold"] / df["close"]).pct_change(21)

    # --- Risk appetite: cyclicals vs defensives (XLY / XLP) ---
    if "xly" in df.columns and "xlp" in df.columns:
        risk_ratio = df["xly"] / df["xlp"]
        f["risk_appetite_change_5d"] = risk_ratio.pct_change(5)
        f["risk_appetite_change_21d"] = risk_ratio.pct_change(21)
        f["risk_appetite_ma_ratio_20"] = risk_ratio / risk_ratio.rolling(20).mean() - 1

    # --- Credit risk appetite: high-yield vs treasuries (HYG / TLT) ---
    if "hyg" in df.columns and "tlt" in df.columns:
        credit_ratio = df["hyg"] / df["tlt"]
        f["credit_change_5d"] = credit_ratio.pct_change(5)
        f["credit_change_21d"] = credit_ratio.pct_change(21)
        f["credit_ma_ratio_20"] = credit_ratio / credit_ratio.rolling(20).mean() - 1

    feature_cols = f.columns.tolist()

    # Forward-fill then zero-fill so a failed macro ticker can't empty the DataFrame.
    # Rows with insufficient price history (e.g. first 252 rows for annual momentum)
    # still get dropped by the dropna below — we only fill NaN that would otherwise
    # persist in ALL rows (i.e. a genuinely missing cross-asset series).
    f[feature_cols] = f[feature_cols].ffill().fillna(0)

    # Targets: shifted -1, so the last row has NaN targets (future unknown)
    f["target_return"] = daily_return.shift(-1)
    f["target_direction"] = (f["target_return"] > 0).astype(float)
    f["close"] = df["close"]

    f = f.dropna(subset=feature_cols)

    return f, feature_cols
