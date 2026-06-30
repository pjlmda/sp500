import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

import config
from src.data import load
from src.features import build


def _sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    if returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))


def _max_drawdown(cumret: pd.Series) -> float:
    peak = cumret.cummax()
    drawdown = (cumret - peak) / peak
    return float(drawdown.min())


def _load_best_params() -> dict:
    path = os.path.join(config.MODELS_DIR, "best_params.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    # Sensible fallback if training hasn't been run yet
    defaults = {"max_depth": 4, "learning_rate": 0.05, "subsample": 0.8,
                "colsample_bytree": 0.7, "min_child_weight": 10, "gamma": 0.1}
    return {"classifier": defaults, "regressor": defaults}


def generate_predictions(min_train_size: int = 500, step: int = 63) -> pd.DataFrame:
    """
    Walk-forward expanding-window predictions, retrained every `step` days.
    Returns raw predictions (date, actual_return, actual_direction, pred_return,
    pred_direction, prob_up) with no strategy/threshold logic applied — that
    happens cheaply downstream in `evaluate()` so thresholds can be swept
    without refitting models.
    """
    df = load()
    feat, feature_cols = build(df)

    data = feat.dropna(subset=["target_return", "target_direction"]).copy()
    n = len(data)
    n_test = n - min_train_size
    n_steps = int(np.ceil(n_test / step))

    best_params = _load_best_params()
    clf_params = best_params["classifier"]
    reg_params = best_params["regressor"]

    print(f"Dataset     : {n} samples  ({data.index[0].date()} to {data.index[-1].date()})")
    print(f"Test period : {n_test} days  |  retraining every {step} days  ({n_steps} fits)")
    print()

    records = []

    for i, train_end in enumerate(range(min_train_size, n, step)):
        train = data.iloc[:train_end]
        test = data.iloc[train_end : min(train_end + step, n)]

        if len(test) == 0:
            break

        X_tr = train[feature_cols].values
        y_reg_tr = train["target_return"].values
        y_clf_tr = train["target_direction"].values.astype(int)
        X_te = test[feature_cols].values

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)

        reg = XGBRegressor(
            n_estimators=300, random_state=42, n_jobs=-1, verbosity=0,
            **reg_params,
        )
        clf = XGBClassifier(
            n_estimators=300, eval_metric="logloss",
            random_state=42, n_jobs=-1, verbosity=0,
            **clf_params,
        )
        reg.fit(X_tr_sc, y_reg_tr)
        clf.fit(X_tr_sc, y_clf_tr)

        pred_returns = reg.predict(X_te_sc)
        pred_dirs = clf.predict(X_te_sc)
        prob_up = clf.predict_proba(X_te_sc)[:, 1]

        for j, (idx, row) in enumerate(test.iterrows()):
            records.append({
                "date": idx,
                "actual_return": row["target_return"],
                "actual_direction": int(row["target_direction"]),
                "pred_return": pred_returns[j],
                "pred_direction": int(pred_dirs[j]),
                "prob_up": prob_up[j],
                "train_size": train_end,
            })

        print(f"  [{i+1:02d}/{n_steps}]  train to {data.index[train_end-1].date()}  "
              f"({train_end} samples)  |  predicted {len(test)} days")

    return pd.DataFrame(records).set_index("date")


def evaluate(
    res: pd.DataFrame,
    confidence_threshold: float = 0.0,
    long_only: bool = True,
) -> tuple[dict, pd.DataFrame]:
    """
    Apply strategy + confidence filtering to raw predictions and compute metrics.

    - confidence_threshold : only trade when |prob_up - 0.5| > threshold.
                              0.0 = always trade (no filter).
                              e.g. 0.05 -> trade only when prob_up > 0.55 or < 0.45
    - long_only            : True -> long or flat; False -> long or short
    """
    res = res.copy()
    prob_up = res["prob_up"]
    confident = (prob_up - 0.5).abs() > confidence_threshold

    if long_only:
        signal = np.where(confident & (prob_up > 0.5), 1.0, 0.0)
    else:
        signal = np.where(confident, np.where(prob_up > 0.5, 1.0, -1.0), 0.0)
    res["signal"] = signal

    res["strategy_return"] = res["actual_return"] * res["signal"]
    res["bnh_return"] = res["actual_return"]
    res["strategy_cumret"] = (1 + res["strategy_return"]).cumprod()
    res["bnh_cumret"] = (1 + res["bnh_return"]).cumprod()

    correct = (res["actual_direction"] == res["pred_direction"]).astype(float)
    res["rolling_accuracy"] = correct.rolling(63).mean()

    traded = res["signal"] != 0
    correct_traded = correct[traded]

    metrics = {
        "direction_accuracy": round(correct.mean(), 4),
        "direction_accuracy_traded": round(correct_traded.mean(), 4) if traded.any() else None,
        "pct_days_traded": round(traded.mean(), 4),
        "confidence_threshold": confidence_threshold,
        "strategy_total_return_pct": round((res["strategy_cumret"].iloc[-1] - 1) * 100, 2),
        "bnh_total_return_pct": round((res["bnh_cumret"].iloc[-1] - 1) * 100, 2),
        "strategy_sharpe": round(_sharpe(res["strategy_return"]), 3),
        "bnh_sharpe": round(_sharpe(res["bnh_return"]), 3),
        "strategy_max_drawdown_pct": round(_max_drawdown(res["strategy_cumret"]) * 100, 2),
        "bnh_max_drawdown_pct": round(_max_drawdown(res["bnh_cumret"]) * 100, 2),
        "n_test_days": len(res),
        "test_start": res.index[0].strftime("%Y-%m-%d"),
        "test_end": res.index[-1].strftime("%Y-%m-%d"),
        "long_only": long_only,
    }
    return metrics, res


def sweep_thresholds(
    raw: pd.DataFrame,
    thresholds: list[float] = (0.0, 0.02, 0.05, 0.08, 0.10, 0.15),
    long_only: bool = True,
) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        m, _ = evaluate(raw, confidence_threshold=t, long_only=long_only)
        rows.append({
            "threshold": t,
            "pct_days_traded": m["pct_days_traded"],
            "accuracy_traded": m["direction_accuracy_traded"],
            "strategy_sharpe": m["strategy_sharpe"],
            "strategy_total_return_pct": m["strategy_total_return_pct"],
            "strategy_max_drawdown_pct": m["strategy_max_drawdown_pct"],
        })
    return pd.DataFrame(rows)


def _print_report(metrics: dict):
    print()
    print("=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)
    print(f"  Confidence threshold: {metrics['confidence_threshold']}")
    print(f"  Days traded         : {metrics['pct_days_traded']:.1%}")
    print(f"  Direction accuracy  : {metrics['direction_accuracy']:.2%}  (all days)")
    if metrics["direction_accuracy_traded"] is not None:
        print(f"  Direction accuracy  : {metrics['direction_accuracy_traded']:.2%}  (traded days only)")
    print()
    print(f"  Strategy  total ret : {metrics['strategy_total_return_pct']:+.1f}%")
    print(f"  B&H       total ret : {metrics['bnh_total_return_pct']:+.1f}%")
    print()
    print(f"  Strategy  Sharpe    : {metrics['strategy_sharpe']:.3f}")
    print(f"  B&H       Sharpe    : {metrics['bnh_sharpe']:.3f}")
    print()
    print(f"  Strategy  max DD    : {metrics['strategy_max_drawdown_pct']:.1f}%")
    print(f"  B&H       max DD    : {metrics['bnh_max_drawdown_pct']:.1f}%")
    print(f"  Test period         : {metrics['test_start']} to {metrics['test_end']}")
    print("=" * 50)


def run(
    min_train_size: int = 500,
    step: int = 63,
    long_only: bool = True,
    confidence_threshold: float = 0.0,
) -> tuple[dict, pd.DataFrame]:
    raw = generate_predictions(min_train_size, step)
    metrics, res = evaluate(raw, confidence_threshold, long_only)
    _print_report(metrics)

    os.makedirs(config.DATA_DIR, exist_ok=True)
    res.to_parquet(os.path.join(config.DATA_DIR, "backtest_results.parquet"))
    with open(os.path.join(config.DATA_DIR, "backtest_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics, res


if __name__ == "__main__":
    run()
