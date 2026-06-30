import os
import json
import joblib
import numpy as np
from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report
from xgboost import XGBClassifier, XGBRegressor

import config
from src.data import load
from src.features import build


def _paths() -> dict:
    return {
        "regressor":   os.path.join(config.MODELS_DIR, "regressor.joblib"),
        "classifier":  os.path.join(config.MODELS_DIR, "classifier.joblib"),
        "scaler":      os.path.join(config.MODELS_DIR, "scaler.joblib"),
        "feature_cols":os.path.join(config.MODELS_DIR, "feature_cols.joblib"),
        "best_params": os.path.join(config.MODELS_DIR, "best_params.json"),
    }


def models_exist() -> bool:
    return all(os.path.exists(p) for p in _paths().values())


def _tune_classifier(X_train, y_train, n_iter: int = 30) -> dict:
    print(f"  Tuning classifier  (RandomizedSearchCV, n_iter={n_iter}, 5-fold TSS)...")
    tscv = TimeSeriesSplit(n_splits=5)
    param_dist = {
        "max_depth":        randint(3, 7),
        "learning_rate":    uniform(0.01, 0.14),
        "subsample":        uniform(0.65, 0.30),
        "colsample_bytree": uniform(0.55, 0.35),
        "min_child_weight": randint(5, 30),
        "gamma":            uniform(0, 0.4),
    }
    base = XGBClassifier(
        n_estimators=300,
        eval_metric="logloss",
        scale_pos_weight=1,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    search = RandomizedSearchCV(
        base, param_dist, n_iter=n_iter, cv=tscv,
        scoring="accuracy", random_state=42, n_jobs=1, verbose=0,
    )
    search.fit(X_train, y_train)
    print(f"  Best CV accuracy: {search.best_score_:.4f}  |  params: {search.best_params_}")
    return search.best_params_


def _tune_regressor(X_train, y_train, n_iter: int = 20) -> dict:
    print(f"  Tuning regressor   (RandomizedSearchCV, n_iter={n_iter}, 5-fold TSS)...")
    tscv = TimeSeriesSplit(n_splits=5)
    param_dist = {
        "max_depth":        randint(3, 7),
        "learning_rate":    uniform(0.01, 0.14),
        "subsample":        uniform(0.65, 0.30),
        "colsample_bytree": uniform(0.55, 0.35),
        "min_child_weight": randint(5, 30),
        "gamma":            uniform(0, 0.4),
    }
    base = XGBRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    search = RandomizedSearchCV(
        base, param_dist, n_iter=n_iter, cv=tscv,
        scoring="neg_mean_absolute_error", random_state=42, n_jobs=1, verbose=0,
    )
    search.fit(X_train, y_train)
    print(f"  Best CV MAE: {-search.best_score_:.6f}  |  params: {search.best_params_}")
    return search.best_params_


def train(force: bool = False) -> dict:
    paths = _paths()

    if not force and models_exist():
        print("Models already trained. Pass force=True to retrain.")
        return {}

    os.makedirs(config.MODELS_DIR, exist_ok=True)

    df = load()
    feat, feature_cols = build(df)
    train_data = feat.dropna(subset=["target_return", "target_direction"])

    X = train_data[feature_cols].values
    y_reg = train_data["target_return"].values
    y_clf = train_data["target_direction"].values.astype(int)

    split = int(len(train_data) * config.TRAIN_RATIO)
    X_train, X_test = X[:split], X[split:]
    y_reg_train, y_reg_test = y_reg[:split], y_reg[split:]
    y_clf_train, y_clf_test = y_clf[:split], y_clf[split:]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    print(f"Training on {split} samples, evaluating on {len(X_test)} samples.")
    print(f"Features: {len(feature_cols)}\n")

    clf_params = _tune_classifier(X_train_sc, y_clf_train)
    reg_params = _tune_regressor(X_train_sc, y_reg_train)

    best_params = {"classifier": clf_params, "regressor": reg_params}

    # Final models with more estimators
    clf = XGBClassifier(
        n_estimators=1000, eval_metric="logloss",
        random_state=42, n_jobs=-1, verbosity=0,
        **clf_params,
    )
    reg = XGBRegressor(
        n_estimators=1000,
        random_state=42, n_jobs=-1, verbosity=0,
        **reg_params,
    )

    print("\nFitting final models (n_estimators=1000)...")
    clf.fit(X_train_sc, y_clf_train)
    reg.fit(X_train_sc, y_reg_train)

    y_clf_pred = clf.predict(X_test_sc)
    y_reg_pred = reg.predict(X_test_sc)

    acc = accuracy_score(y_clf_test, y_clf_pred)
    mae = mean_absolute_error(y_reg_test, y_reg_pred)

    print(f"\nTest set size : {len(X_test)} samples")
    print(f"Classifier Acc: {acc:.4f}")
    print(f"Regressor MAE : {mae:.6f}  (daily return units)")
    print(classification_report(y_clf_test, y_clf_pred, target_names=["Down", "Up"]))

    joblib.dump(reg, paths["regressor"])
    joblib.dump(clf, paths["classifier"])
    joblib.dump(scaler, paths["scaler"])
    joblib.dump(feature_cols, paths["feature_cols"])
    with open(paths["best_params"], "w") as f:
        json.dump(best_params, f, indent=2)

    print("Models saved.")
    return {"accuracy": acc, "mae": mae, "test_samples": len(X_test)}


if __name__ == "__main__":
    train(force=True)
