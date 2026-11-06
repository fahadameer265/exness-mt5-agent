"""Walk-forward training for one symbol and one direction.

The model estimates P(the 1:2 target prints before the stop within 12 bars |
this setup). It is fit only on setup rows. Cross-validation reports precision
at the live threshold and does not pick that threshold from the test folds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from exness_mt5_agent.config import StrategyParams
from exness_mt5_agent.features import FEATURE_COLUMNS
from exness_mt5_agent.validation import precision_at_threshold, purged_folds


@dataclass
class FoldScore:
    fold: int
    train_rows: int
    test_rows: int
    base_rate: float
    precision: float | None
    signals: int


@dataclass
class TrainResult:
    model: object
    rows: int
    positives: int
    folds: list[FoldScore]
    holdout_precision: float | None
    holdout_signals: int
    holdout_base_rate: float
    holdout_rows: int


def _estimator(model_type: str, scale_pos_weight: float, n_estimators: int):
    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=5,
            min_samples_leaf=20,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
    if model_type != "xgboost":
        raise ValueError("model_type must be xgboost or random_forest.")
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )


def _fit(model, x_train, y_train, x_valid=None, y_valid=None):
    if x_valid is None or model.__class__.__name__ != "XGBClassifier":
        model.fit(x_train, y_train)
        return model
    try:
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            verbose=False,
            early_stopping_rounds=30,
        )
        return model
    except TypeError:
        model.set_params(early_stopping_rounds=30)
        model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
        return model


def _best_tree_count(model, fallback: int) -> int:
    best = getattr(model, "best_iteration", None)
    if best is None:
        return fallback
    return int(best) + 1


def _positive_weight(y: pd.Series) -> float:
    positives = float((y == 1).sum())
    negatives = float((y == 0).sum())
    if positives <= 0 or negatives <= 0:
        raise ValueError("A training side needs both wins and losses.")
    return negatives / positives


def _xy(frame: pd.DataFrame, label_col: str) -> tuple[pd.DataFrame, pd.Series]:
    missing = [col for col in FEATURE_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"Feature frame is missing {missing}.")
    x = frame.loc[:, FEATURE_COLUMNS].astype(float)
    y = frame[label_col].astype(int)
    return x, y


def train_direction(
    setups: pd.DataFrame,
    label_col: str,
    params: StrategyParams,
    model_type: str,
    holdout_frac: float = 0.15,
    min_rows: int = 200,
) -> TrainResult:
    """Fit one direction on the earlier setups and score the later holdout."""

    frame = setups.dropna(subset=FEATURE_COLUMNS + [label_col, "bar_index"]).copy()
    frame = frame.sort_values("bar_index")
    if len(frame) < min_rows:
        raise ValueError(f"Only {len(frame)} labeled setups; need at least {min_rows}.")
    cutoff_pos = int(len(frame) * (1.0 - holdout_frac))
    cutoff_pos = min(max(cutoff_pos, min_rows // 2), len(frame) - 1)
    hold_start = int(frame["bar_index"].iloc[cutoff_pos])
    train_frame = frame.loc[frame["bar_index"] < hold_start - params.embargo_bars].copy()
    hold_frame = frame.iloc[cutoff_pos:].copy()
    if len(train_frame) < min_rows // 2 or len(hold_frame) < 20:
        raise ValueError("Holdout split left too little train or test data.")

    folds: list[FoldScore] = []
    x_all, y_all = _xy(train_frame, label_col)
    bar_index = train_frame["bar_index"].to_numpy()
    for fold_id, (train_pos, test_pos) in enumerate(
        purged_folds(bar_index, params.n_splits, params.embargo_bars), start=1
    ):
        y_train = y_all.iloc[train_pos]
        y_test = y_all.iloc[test_pos]
        if y_train.nunique() < 2 or y_test.nunique() < 1:
            continue
        model = _estimator(model_type, _positive_weight(y_train), n_estimators=300)
        _fit(model, x_all.iloc[train_pos], y_train)
        probability = model.predict_proba(x_all.iloc[test_pos])[:, 1]
        precision, signals = precision_at_threshold(
            y_test.to_numpy(), probability, params.proba_threshold
        )
        folds.append(
            FoldScore(
                fold=fold_id,
                train_rows=len(train_pos),
                test_rows=len(test_pos),
                base_rate=float(y_test.mean()),
                precision=precision,
                signals=signals,
            )
        )
    if not folds:
        raise ValueError("Every walk-forward fold was skipped. Need a longer history.")

    # Choose the tree count on a trailing slice, then refit on all pre-holdout rows.
    tail = max(30, int(len(train_frame) * 0.15))
    core = train_frame.iloc[:-tail]
    valid = train_frame.iloc[-tail:]
    if core[label_col].nunique() < 2:
        core = train_frame
        valid = None
    probe = _estimator(model_type, _positive_weight(core[label_col]), n_estimators=300)
    if valid is not None and valid[label_col].nunique() == 2:
        x_core, y_core = _xy(core, label_col)
        x_valid, y_valid = _xy(valid, label_col)
        _fit(probe, x_core, y_core, x_valid, y_valid)
        n_estimators = _best_tree_count(probe, 300)
    else:
        n_estimators = 300
    final = _estimator(model_type, _positive_weight(train_frame[label_col]), n_estimators)
    x_train, y_train = _xy(train_frame, label_col)
    _fit(final, x_train, y_train)

    x_hold, y_hold = _xy(hold_frame, label_col)
    hold_probability = final.predict_proba(x_hold)[:, 1]
    hold_precision, hold_signals = precision_at_threshold(
        y_hold.to_numpy(), hold_probability, params.proba_threshold
    )
    return TrainResult(
        model=final,
        rows=len(train_frame),
        positives=int((y_train == 1).sum()),
        folds=folds,
        holdout_precision=hold_precision,
        holdout_signals=hold_signals,
        holdout_base_rate=float(y_hold.mean()),
        holdout_rows=len(hold_frame),
    )


def predict_proba(model, row: pd.DataFrame) -> float:
    """Probability of class 1 for a one-row feature frame."""

    probability = model.predict_proba(row.loc[:, FEATURE_COLUMNS].astype(float))[:, 1]
    return float(probability[0])
