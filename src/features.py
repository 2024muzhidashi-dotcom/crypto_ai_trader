from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def add_common_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy().sort_values("timestamp").reset_index(drop=True)
    data["ema20"] = ema(data["close"], 20)
    data["ema50"] = ema(data["close"], 50)
    data["ema100"] = ema(data["close"], 100)
    data["ma20"] = sma(data["close"], 20)
    data["ma55"] = sma(data["close"], 55)
    data["ma100"] = sma(data["close"], 100)
    data["rsi14"] = rsi(data["close"], 14)
    data["atr14"] = atr(data, 14)
    data["return"] = data["close"].pct_change()
    data["volume_change"] = data["volume"].pct_change()
    return data.dropna().reset_index(drop=True)


def merge_timeframe_features(anchor_df: pd.DataFrame, higher_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    cols = [col for col in higher_df.columns if col != "timestamp"]
    renamed = higher_df[["timestamp", *cols]].rename(columns={col: f"{prefix}_{col}" for col in cols})
    return pd.merge_asof(anchor_df.sort_values("timestamp"), renamed.sort_values("timestamp"), on="timestamp", direction="backward")


def build_trade_dataset(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Clean restart: keep only merged data as-is.
    New strategy-specific feature engineering will be rebuilt from scratch.
    """

    return merged.copy().sort_values("timestamp").reset_index(drop=True)
