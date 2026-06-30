# Feature Reference

All features are computed in `src/features.py` via `build(df)` and fed into the XGBoost classifier (direction) and regressor (return). The function returns a `(DataFrame, feature_cols)` tuple; the last row always has valid features but NaN targets, which is the inference point used by `predict_next_day()`.

---

## 1. Short-term lagged returns

| Feature | Formula | Intuition |
|---|---|---|
| `return_lag_1` | `close.pct_change().shift(1)` | Yesterday's return — captures mean-reversion or momentum at the 1-day horizon |
| `return_lag_2` | `close.pct_change().shift(2)` | 2 days ago |
| `return_lag_3` | `close.pct_change().shift(3)` | 3 days ago |
| `return_lag_5` | `close.pct_change().shift(5)` | 1 week ago |
| `return_lag_10` | `close.pct_change().shift(10)` | 2 weeks ago |

Lags 1–10 give the model a short memory of recent price moves. Positive autocorrelation in this window = trend; negative = mean reversion.

---

## 2. Longer-term momentum

| Feature | Formula | Intuition |
|---|---|---|
| `return_lag_21` | `close.pct_change(21)` | 1-month return — short-term momentum |
| `return_lag_63` | `close.pct_change(63)` | 1-quarter return |
| `return_lag_126` | `close.pct_change(126)` | 6-month return |
| `return_lag_252` | `close.pct_change(252)` | 1-year return |
| `momentum_12_1` | `return_lag_252 - return_lag_21` | Classic Jegadeesh-Titman factor: 12-month return excluding the most recent month. Skips the last month to avoid short-term reversal noise. Positive = sustained medium-term trend. |

---

## 3. Moving average ratios

| Feature | Formula | Intuition |
|---|---|---|
| `ma_ratio_5` | `close / MA(5) - 1` | Price deviation from 5-day MA — very short trend |
| `ma_ratio_10` | `close / MA(10) - 1` | Price deviation from 2-week MA |
| `ma_ratio_20` | `close / MA(20) - 1` | Price deviation from 1-month MA |
| `ma_ratio_50` | `close / MA(50) - 1` | Price deviation from ~2-month MA — medium-term trend |

Positive = price above its moving average (trend following signal). Negative = price below (potential mean-reversion or continuation of downtrend).

---

## 4. Volatility

| Feature | Formula | Intuition |
|---|---|---|
| `volatility_5` | Rolling 5-day std of daily returns | Short-term volatility regime — spikes often precede or follow large moves |
| `volatility_20` | Rolling 20-day std of daily returns | Monthly volatility — captures regime shifts (low vol → complacency, high vol → fear) |

---

## 5. RSI (Relative Strength Index)

| Feature | Formula | Intuition |
|---|---|---|
| `rsi` | Wilder's RSI(14) on close price | Oscillator between 0–100. >70 = overbought (potential reversal), <30 = oversold. Uses exponential smoothing (EWM) on gains and losses. |

Construction: `RS = EWM_avg_gain / EWM_avg_loss`, then `RSI = 100 - 100/(1+RS)`.

---

## 6. MACD (Moving Average Convergence Divergence)

| Feature | Formula | Intuition |
|---|---|---|
| `macd` | `(EMA12 - EMA26) / close` | Normalised difference between fast and slow EMAs — trend direction and strength |
| `macd_signal` | `EMA9(macd)` | Smoothed MACD line — the "signal line" crossovers are classic buy/sell triggers |
| `macd_hist` | `macd - macd_signal` | Histogram of convergence/divergence — positive = momentum building, negative = fading |

Normalised by close price to make the feature stationary across different price levels.

---

## 7. Bollinger Bands

| Feature | Formula | Intuition |
|---|---|---|
| `bb_pct_b` | `(close - lower) / (upper - lower)` | Position within the bands (0 = at lower band, 1 = at upper). >1 or <0 = breakout. Used as a mean-reversion signal. |
| `bb_width` | `(upper - lower) / mid` | Band width normalised by the mid-band. Low width = low volatility / potential squeeze. High width = expanding volatility. |

Bands use a 20-day rolling window with ±2 standard deviations.

---

## 8. Volume ratios

| Feature | Formula | Intuition |
|---|---|---|
| `volume_ratio_5` | `volume / MA_volume(5)` | Volume vs its 5-day average. >1 = unusually high activity — can confirm or question a price move |
| `volume_ratio_20` | `volume / MA_volume(20)` | Volume vs 1-month average — captures unusual participation at a monthly scale |

---

## 9. Price action

| Feature | Formula | Intuition |
|---|---|---|
| `hl_range` | `(high - low) / close` | Intraday range normalised by price — proxy for intraday volatility and uncertainty |
| `gap` | `(open - prev_close) / prev_close` | Overnight gap — captures news and pre-market sentiment. Positive gap = bullish open. |

---

## 10. Calendar

| Feature | Formula | Intuition |
|---|---|---|
| `day_of_week` | `0=Monday … 4=Friday` | Day-of-week effect (e.g. Monday effect, Friday positioning) |
| `month` | `1–12` | Seasonal patterns (e.g. January effect, sell-in-May) |

Note: these have low feature importance in practice; the model mostly ignores them.

---

## 11. VIX (CBOE Volatility Index)

Source: `^VIX` via yfinance.

| Feature | Formula | Intuition |
|---|---|---|
| `vix_level` | Raw VIX value | Absolute fear level. >30 = high fear, <15 = complacency. |
| `vix_change_1d` | `vix.pct_change()` | Daily VIX move — spike = sudden fear event |
| `vix_change_5d` | `vix.pct_change(5)` | Weekly VIX trend |
| `vix_ma_ratio_10` | `vix / MA_vix(10) - 1` | VIX relative to its recent average — mean reversion in fear |
| `vix_zscore_20` | `(vix - mean_vix_20) / std_vix_20` | Normalised VIX level over 20 days. High z-score = fear spike above recent norms. |

---

## 12. Yield curve (10Y − 3M)

Sources: `^TNX` (10Y Treasury yield) and `^IRX` (3-month T-bill yield) via yfinance.

| Feature | Formula | Intuition |
|---|---|---|
| `yield_curve` | `TNX - IRX` (in %) | Spread between long and short rates. Negative = inverted curve, historically precedes recessions. Positive and rising = healthy expansion expectations. |
| `yield_curve_change_5d` | `yield_curve.diff(5)` | Weekly change in the spread — steepening vs flattening signal |
| `yield_curve_change_21d` | `yield_curve.diff(21)` | Monthly change — medium-term regime shift |
| `yield_curve_zscore_63` | `(yc - mean_yc_63) / std_yc_63` | Normalised spread vs its quarterly mean. Captures how unusual current curve shape is relative to recent history. |

---

## 13. Dollar strength

Source: `UUP` (Invesco DB US Dollar Index Bullish Fund ETF) via yfinance.

| Feature | Formula | Intuition |
|---|---|---|
| `dollar_change_1d` | `UUP.pct_change()` | Daily USD move. Strong dollar = risk-off pressure on equities (tighter global financial conditions). |
| `dollar_change_5d` | `UUP.pct_change(5)` | Weekly dollar trend |
| `dollar_ma_ratio_20` | `UUP / MA_UUP(20) - 1` | Dollar strength relative to its 1-month average |

---

## 14. Gold (safe-haven flows)

Source: `GLD` (SPDR Gold Shares ETF) via yfinance.

| Feature | Formula | Intuition |
|---|---|---|
| `gold_change_1d` | `GLD.pct_change()` | Daily gold move. Gold rising = flight-to-safety, often negative for equities. |
| `gold_change_5d` | `GLD.pct_change(5)` | Weekly gold trend |
| `gold_ma_ratio_20` | `GLD / MA_GLD(20) - 1` | Gold relative to its 1-month average |
| `gold_spx_ratio_change_21d` | `(GLD/SPX).pct_change(21)` | Monthly change in the Gold/S&P ratio. Rising = gold outperforming equities = risk-off rotation. |

---

## 15. Risk appetite — cyclicals vs defensives

Sources: `XLY` (Consumer Discretionary) and `XLP` (Consumer Staples) sector ETFs via yfinance.

| Feature | Formula | Intuition |
|---|---|---|
| `risk_appetite_change_5d` | `(XLY/XLP).pct_change(5)` | Weekly move in the cyclicals-to-defensives ratio. Rising = investors rotating into risk. Falling = defensive positioning. |
| `risk_appetite_change_21d` | `(XLY/XLP).pct_change(21)` | Monthly trend in risk appetite |
| `risk_appetite_ma_ratio_20` | `(XLY/XLP) / MA(XLY/XLP, 20) - 1` | Ratio relative to its recent average |

XLY/XLP is a widely used institutional risk-on/risk-off indicator: when investors are bullish they buy discretionary (XLY), when bearish they rotate to staples (XLP).

---

## 16. Credit risk appetite — high-yield vs treasuries

Sources: `HYG` (iShares iBoxx High Yield Corporate Bond ETF) and `TLT` (iShares 20+ Year Treasury Bond ETF) via yfinance.

| Feature | Formula | Intuition |
|---|---|---|
| `credit_change_5d` | `(HYG/TLT).pct_change(5)` | Weekly credit risk appetite. Falling = credit spreads widening, stress building — typically precedes equity weakness. |
| `credit_change_21d` | `(HYG/TLT).pct_change(21)` | Monthly credit trend |
| `credit_ma_ratio_20` | `(HYG/TLT) / MA(HYG/TLT, 20) - 1` | Credit ratio relative to recent average |

HYG/TLT captures the risk premium demanded by credit markets. When this ratio falls, high-yield spreads are widening — a leading indicator of equity stress.

---

## Targets (not model inputs)

| Column | Formula | Used for |
|---|---|---|
| `target_return` | `daily_return.shift(-1)` | Regressor label — next day's return |
| `target_direction` | `1 if target_return > 0 else 0` | Classifier label — next day's direction |

The last row in the feature DataFrame always has `NaN` targets (the future is unknown) and is the row used for next-day inference in `src/predict.py`.
