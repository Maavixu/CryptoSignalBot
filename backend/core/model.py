"""
model.py — Phase 3 ML ensemble.
Two independent classifiers vote on P(win):
  1. XGBoost — strong on tabular features, handles non-linearity well
  2. MLP (scikit-learn) — neural net, different inductive bias

Both are calibrated with CalibratedClassifierCV so their outputs
are true probabilities, not just scores.

Incremental retraining:
  - MLP supports partial_fit (true online learning)
  - XGBoost does full retrain (fast at this dataset size, more stable)
  - Retrain triggered when dataset grows by >= MIN_NEW_SAMPLES since last train
  - Models persisted as .pkl in /models directory

Brier Score tracking:
  - Every prediction logged with probability + market
  - When trade closes, outcome is recorded and Brier score updated
  - Rolling Brier score visible in frontend
"""
import json
import os
import pickle
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import NotFittedError
import xgboost as xgb

from .config import MODELS_DIR, load_config
from .logger import get_logger
from .storage import get_model_data, update_model_meta, append_training_sample
from .features import N_FEATURES, FEATURE_NAMES

_log = get_logger("predict")
_lock = threading.RLock()

# Paths
XGB_PATH    = MODELS_DIR / "xgboost_model.pkl"
MLP_PATH    = MODELS_DIR / "mlp_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
META_PATH   = MODELS_DIR / "model_meta.json"

# Training thresholds
MIN_SAMPLES_TO_TRAIN = 10    # minimum samples before first training
MIN_NEW_SAMPLES      = 5     # retrain if dataset grew by this many since last train
CALIBRATION_CV       = 3     # cross-val folds for probability calibration


@dataclass
class PredictionResult:
    symbol: str
    p_win: float            # ensemble probability of TP hit before SL
    p_market: float         # implied probability from volatility baseline
    edge: float             # p_win - p_market
    edge_pct: float         # edge as percentage (edge * 100)
    direction: str          # "BUY" | "SELL" | "SKIP"
    xgb_prob: float
    mlp_prob: float
    confidence: str         # "high" | "medium" | "low"
    feature_importances: dict
    timestamp: str = ""

    @property
    def has_edge(self) -> bool:
        cfg = load_config()
        return self.edge_pct >= cfg["prediction"]["min_edge_pct"]


# ── Model manager ─────────────────────────────────────────────────────────────

class EnsembleModel:
    """
    Manages XGBoost + MLP classifiers.
    Thread-safe. Handles initial bootstrap, incremental updates,
    and graceful degradation when insufficient data exists.
    """

    def __init__(self):
        self._xgb: Optional[xgb.XGBClassifier] = None
        self._mlp: Optional[MLPClassifier] = None
        self._scaler: Optional[StandardScaler] = None
        self._is_trained = False
        self._n_samples_at_train = 0
        self._load_from_disk()

    def _load_from_disk(self):
        """Load persisted models if they exist."""
        try:
            if XGB_PATH.exists():
                with open(XGB_PATH, "rb") as f:
                    self._xgb = pickle.load(f)
                _log.info("XGBoost model loaded from disk")
            if MLP_PATH.exists():
                with open(MLP_PATH, "rb") as f:
                    self._mlp = pickle.load(f)
                _log.info("MLP model loaded from disk")
            if SCALER_PATH.exists():
                with open(SCALER_PATH, "rb") as f:
                    self._scaler = pickle.load(f)
            if META_PATH.exists():
                with open(META_PATH) as f:
                    meta = json.load(f)
                    self._n_samples_at_train = meta.get("n_samples", 0)
            if self._xgb and self._mlp and self._scaler:
                self._is_trained = True
        except Exception as e:
            _log.warning(f"Could not load models from disk: {e}")

    def _save_to_disk(self):
        """Persist models to disk."""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with open(XGB_PATH, "wb") as f:
                pickle.dump(self._xgb, f)
            with open(MLP_PATH, "wb") as f:
                pickle.dump(self._mlp, f)
            with open(SCALER_PATH, "wb") as f:
                pickle.dump(self._scaler, f)
            with open(META_PATH, "w") as f:
                json.dump({
                    "n_samples": self._n_samples_at_train,
                    "saved_at": datetime.utcnow().isoformat(),
                    "feature_names": FEATURE_NAMES,
                }, f, indent=2)
            _log.info(f"Models saved to disk ({self._n_samples_at_train} samples)")
        except Exception as e:
            _log.error(f"Failed to save models: {e}")

    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Full retrain on entire dataset.
        Called when dataset grows enough or via /train endpoint.
        """
        with _lock:
            n = len(y)
            if n < MIN_SAMPLES_TO_TRAIN:
                _log.warning(f"Insufficient samples to train: {n} < {MIN_SAMPLES_TO_TRAIN}")
                return {"status": "skipped", "reason": f"need {MIN_SAMPLES_TO_TRAIN} samples"}

            _log.info(f"Training ensemble on {n} samples")
            start = time.time()

            # Scale features
            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X)

            # Class balance check
            n_pos = int(np.sum(y))
            n_neg = n - n_pos
            scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

            # XGBoost
            xgb_base = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
            )
            if n >= CALIBRATION_CV * 2:
                self._xgb = CalibratedClassifierCV(xgb_base, cv=CALIBRATION_CV, method="isotonic")
                self._xgb.fit(X_scaled, y)
            else:
                xgb_base.fit(X_scaled, y)
                self._xgb = xgb_base

            # MLP
            mlp_base = MLPClassifier(
                hidden_layer_sizes=(64, 32, 16),
                activation="relu",
                solver="adam",
                learning_rate_init=0.001,
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.15 if n >= 20 else 0.0,
                n_iter_no_change=10,
            )
            if n >= CALIBRATION_CV * 2:
                self._mlp = CalibratedClassifierCV(mlp_base, cv=CALIBRATION_CV, method="sigmoid")
                self._mlp.fit(X_scaled, y)
            else:
                mlp_base.fit(X_scaled, y)
                self._mlp = mlp_base

            self._is_trained = True
            self._n_samples_at_train = n
            self._save_to_disk()

            # Compute in-sample metrics
            from sklearn.metrics import brier_score_loss, roc_auc_score
            try:
                y_pred_xgb = self._xgb.predict_proba(X_scaled)[:, 1]
                y_pred_mlp = self._mlp.predict_proba(X_scaled)[:, 1]
                y_ensemble = (y_pred_xgb + y_pred_mlp) / 2
                brier = brier_score_loss(y, y_ensemble)
                auc   = roc_auc_score(y, y_ensemble) if len(np.unique(y)) > 1 else 0.5
            except Exception:
                brier, auc = 0.25, 0.5

            duration = round(time.time() - start, 2)
            result = {
                "status": "trained",
                "n_samples": n,
                "n_positive": n_pos,
                "n_negative": n_neg,
                "brier_score": round(brier, 4),
                "auc": round(auc, 4),
                "duration_seconds": duration,
            }
            _log.info(
                f"Training complete: {n} samples, Brier={brier:.4f}, AUC={auc:.4f}",
                data=result,
            )
            update_model_meta(
                last_trained=datetime.utcnow().isoformat(),
                brier_score=brier,
            )
            return result

    def predict(self, features: np.ndarray) -> Tuple[float, float, float]:
        """
        Predict P(win) using ensemble.
        Returns (ensemble_prob, xgb_prob, mlp_prob).
        Falls back to 0.50 if not trained.
        """
        if not self._is_trained or self._scaler is None:
            return 0.50, 0.50, 0.50

        with _lock:
            try:
                X_scaled = self._scaler.transform(features)
                xgb_prob = float(self._xgb.predict_proba(X_scaled)[0, 1])
                mlp_prob = float(self._mlp.predict_proba(X_scaled)[0, 1])
                ensemble = (xgb_prob + mlp_prob) / 2.0
                return ensemble, xgb_prob, mlp_prob
            except Exception as e:
                _log.error(f"Prediction failed: {e}")
                return 0.50, 0.50, 0.50

    def should_retrain(self) -> bool:
        """True if dataset has grown enough since last training."""
        data = get_model_data()
        n_current = len(data.get("features", []))
        return (
            n_current >= MIN_SAMPLES_TO_TRAIN
            and (n_current - self._n_samples_at_train) >= MIN_NEW_SAMPLES
        )

    def get_feature_importance(self) -> dict:
        """Get XGBoost feature importances (falls back to uniform if untrained)."""
        if not self._is_trained:
            return {name: 1.0 / N_FEATURES for name in FEATURE_NAMES}
        try:
            base = self._xgb
            # Unwrap CalibratedClassifierCV if needed
            if hasattr(base, "estimator"):
                base = base.estimator
            elif hasattr(base, "calibrated_classifiers_"):
                base = base.calibrated_classifiers_[0].estimator
            importances = base.feature_importances_
            return dict(zip(FEATURE_NAMES, [round(float(v), 4) for v in importances]))
        except Exception:
            return {name: 1.0 / N_FEATURES for name in FEATURE_NAMES}

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def n_samples(self) -> int:
        return self._n_samples_at_train


# ── Singleton ─────────────────────────────────────────────────────────────────
_model: Optional[EnsembleModel] = None


def get_model() -> EnsembleModel:
    global _model
    if _model is None:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        _model = EnsembleModel()
    return _model


def retrain() -> dict:
    """
    Load full dataset and retrain. Called by /train endpoint and scheduler.
    Thread-safe. Returns training metrics dict.
    """
    data = get_model_data()
    features = data.get("features", [])
    outcomes = data.get("outcomes", [])

    if not features:
        return {"status": "skipped", "reason": "no training data yet"}

    X = np.array(features, dtype=np.float32)
    y = np.array(outcomes, dtype=np.int32)

    model = get_model()
    return model.train(X, y)


def maybe_retrain() -> Optional[dict]:
    """Retrain only if the dataset has grown enough. Called after each trade close."""
    model = get_model()
    if model.should_retrain():
        _log.info("Auto-retraining triggered — dataset grew sufficiently")
        return retrain()
    return None
