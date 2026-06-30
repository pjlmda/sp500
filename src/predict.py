import joblib
import numpy as np
import os

import config
from src.data import load
from src.features import build


def _load_artifacts() -> tuple:
    paths = {
        "regressor": os.path.join(config.MODELS_DIR, "regressor.joblib"),
        "classifier": os.path.join(config.MODELS_DIR, "classifier.joblib"),
        "scaler": os.path.join(config.MODELS_DIR, "scaler.joblib"),
        "feature_cols": os.path.join(config.MODELS_DIR, "feature_cols.joblib"),
    }
    missing = [k for k, p in paths.items() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Missing model artifacts: {missing}. Run src/train.py first."
        )
    reg = joblib.load(paths["regressor"])
    clf = joblib.load(paths["classifier"])
    scaler = joblib.load(paths["scaler"])
    feature_cols = joblib.load(paths["feature_cols"])
    return reg, clf, scaler, feature_cols


def predict_next_day(force_refresh: bool = False) -> dict:
    """
    Returns a prediction dict for the next trading day based on the latest data.
    """
    df = load(force_refresh=force_refresh)
    feat, _ = build(df)
    reg, clf, scaler, feature_cols = _load_artifacts()

    if feat.empty:
        raise RuntimeError(
            "Feature DataFrame is empty after build() — one or more data series "
            "likely failed to download. Check yfinance connectivity."
        )

    # Last row has valid features but NaN targets — that's our inference point
    last = feat[feature_cols].iloc[[-1]].values
    last_sc = scaler.transform(last)

    pred_return = reg.predict(last_sc)[0]
    pred_direction = clf.predict(last_sc)[0]
    pred_proba = clf.predict_proba(last_sc)[0]

    last_close = feat["close"].iloc[-1]
    pred_price = last_close * (1 + pred_return)

    return {
        "as_of_date": feat.index[-1].strftime("%Y-%m-%d"),
        "last_close": round(float(last_close), 2),
        "predicted_price": round(float(pred_price), 2),
        "predicted_return_pct": round(float(pred_return) * 100, 4),
        "predicted_direction": "Up" if pred_direction == 1 else "Down",
        "direction_confidence_pct": round(float(np.max(pred_proba)) * 100, 2),
        "prob_up": round(float(pred_proba[1]) * 100, 2),
        "prob_down": round(float(pred_proba[0]) * 100, 2),
    }


if __name__ == "__main__":
    result = predict_next_day()
    for k, v in result.items():
        print(f"{k:30s}: {v}")
