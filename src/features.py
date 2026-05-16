"""Technical indicator computation and feature engineering."""
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Raw indicator helpers
# ---------------------------------------------------------------------------

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def _stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14, d: int = 3
) -> tuple[pd.Series, pd.Series]:
    lowest = low.rolling(k).min()
    highest = high.rolling(k).max()
    stoch_k = 100 * (close - lowest) / (highest - lowest).replace(0, np.nan)
    stoch_d = stoch_k.rolling(d).mean()
    return stoch_k, stoch_d


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


# ---------------------------------------------------------------------------
# Indicator layer
# ---------------------------------------------------------------------------

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical analysis columns to a copy of the OHLCV DataFrame."""
    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # Returns
    for n in [1, 3, 5, 10, 20]:
        df[f"return_{n}d"] = close.pct_change(n)

    # Moving averages
    for n in [10, 20, 50, 200]:
        df[f"sma_{n}"] = close.rolling(n).mean()
    df["ema_12"] = close.ewm(span=12, adjust=False).mean()
    df["ema_26"] = close.ewm(span=26, adjust=False).mean()

    # Price / MA ratios
    df["close_sma20_ratio"] = close / df["sma_20"]
    df["close_sma50_ratio"] = close / df["sma_50"]
    df["sma20_sma50_ratio"] = df["sma_20"] / df["sma_50"]

    # RSI
    df["rsi_14"] = _rsi(close, 14)

    # MACD
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Bollinger Bands (20, 2σ)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / bb_mid
    bb_range = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
    df["bb_pct"] = (close - df["bb_lower"]) / bb_range

    # ATR (normalised)
    df["atr_14"] = _atr(high, low, close, 14)
    df["atr_ratio"] = df["atr_14"] / close.replace(0, np.nan)

    # Stochastic
    df["stoch_k"], df["stoch_d"] = _stochastic(high, low, close, 14, 3)

    # Volume
    vol_sma = volume.rolling(20).mean().replace(0, np.nan)
    df["volume_ratio"] = volume / vol_sma
    df["obv"] = _obv(close, volume)
    df["obv_sma20"] = df["obv"].rolling(20).mean()
    df["obv_ratio"] = df["obv"] / df["obv_sma20"].replace(0, np.nan)

    # Historical volatility
    df["volatility_20d"] = close.pct_change().rolling(20).std()

    # Lagged 1-day returns (momentum memory)
    for lag in [1, 2, 3, 5]:
        df[f"lag_return_{lag}d"] = df["return_1d"].shift(lag)

    return df


# ---------------------------------------------------------------------------
# Feature + target engineering
# ---------------------------------------------------------------------------

FEATURE_COLS: list[str] = [
    "return_1d", "return_3d", "return_5d", "return_10d", "return_20d",
    "close_sma20_ratio", "close_sma50_ratio", "sma20_sma50_ratio",
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "bb_width", "bb_pct",
    "atr_ratio",
    "stoch_k", "stoch_d",
    "volume_ratio",
    "obv_ratio",
    "volatility_20d",
    "lag_return_1d", "lag_return_2d", "lag_return_3d", "lag_return_5d",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add indicators + target column and drop rows with any NaN.

    Target added:
      fwd_close – next-day close (regression)
    """
    df = add_technical_indicators(df)
    df["fwd_close"] = df["Close"].shift(-1)
    return df.dropna(subset=FEATURE_COLS + ["fwd_close"])


def get_latest_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a single-row DataFrame of features for the most recent complete bar.
    Used for live prediction (no forward return required).
    """
    df = add_technical_indicators(df)
    valid = df[FEATURE_COLS].dropna()
    if valid.empty:
        raise ValueError("Not enough data to compute features.")
    return valid.iloc[[-1]]
