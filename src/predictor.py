"""XGBoost next-day price predictor."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

import Stock_Analyzer.config as config
from Stock_Analyzer.src.features import FEATURE_COLS, build_features, get_latest_features


class PricePredictor:
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.model = xgb.XGBRegressor(
            n_estimators=config.N_ESTIMATORS,
            max_depth=config.MAX_DEPTH,
            learning_rate=config.LEARNING_RATE,
            subsample=config.SUBSAMPLE,
            colsample_bytree=config.COLSAMPLE_BYTREE,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )
        self.feature_cols: list[str] = FEATURE_COLS
        self.trained: bool = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> dict:
        """Train on historical OHLCV data. Returns evaluation metrics."""
        data = build_features(df)
        X = data[self.feature_cols]
        y = data["fwd_close"]

        split_idx = int(len(X) * (1 - config.TEST_SIZE))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        self.model.fit(X_train, y_train)
        self.trained = True

        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mape = float(np.mean(np.abs((y_test.values - y_pred) / y_test.values)) * 100)

        return {
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "n_train": len(X_train),
            "n_test": len(X_test),
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_next(self, df: pd.DataFrame) -> float:
        """Predict next-day close price from the most recent (live) bar."""
        self._assert_trained()
        X = get_latest_features(df)
        return float(self.model.predict(X)[0])

    def predict_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return predicted vs actual next-day close for the full history.
        Useful for backtesting visualisation.
        """
        self._assert_trained()
        data = build_features(df)
        X = data[self.feature_cols]
        preds = self.model.predict(X)

        out = data[["Close", "fwd_close"]].copy()
        out["predicted_close"] = preds
        out["pred_error_pct"] = (out["predicted_close"] - out["fwd_close"]) / out["fwd_close"] * 100
        return out

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def save(self, path: Path | None = None) -> None:
        if path is None:
            path = config.MODELS_DIR / f"{self.ticker}_predictor.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(ticker: str, path: Path | None = None) -> "PricePredictor":
        if path is None:
            path = config.MODELS_DIR / f"{ticker}_predictor.pkl"
        return joblib.load(path)

    def _assert_trained(self) -> None:
        if not self.trained:
            raise RuntimeError(f"[{self.ticker}] Model not trained. Call fit() first.")
