from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import ccxt
import pandas as pd

from .config import settings
from .utils import symbol_to_filename


@dataclass
class DownloadTask:
    symbol: str
    timeframe: str
    months: int


class OKXPerpDataLoader:
    def __init__(self, exchange_id: str = settings.exchange_id) -> None:
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class({"enableRateLimit": True, "options": {"defaultType": "swap"}})
        self.exchange.load_markets()

    def discover_btc_perpetuals(self, base_asset: str = settings.base_asset) -> list[str]:
        symbols: list[str] = []
        for symbol, market in self.exchange.markets.items():
            if market.get("swap") and market.get("base") == base_asset:
                symbols.append(symbol)
        return sorted(set(symbols))

    def fetch_ohlcv_history(self, symbol: str, timeframe: str, months: int, limit: int = 200) -> pd.DataFrame:
        since = datetime.now(timezone.utc) - timedelta(days=months * 30)
        cursor = int(since.timestamp() * 1000)
        rows: list[list[float]] = []
        requests_count = 0
        while True:
            batch = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=limit)
            if not batch:
                break
            rows.extend(batch)
            requests_count += 1
            if requests_count % 100 == 0:
                print(f"{symbol} {timeframe}: fetched {len(rows)} rows so far...")
            last_ts = batch[-1][0]
            next_cursor = last_ts + 1
            if next_cursor <= cursor or len(batch) < limit:
                break
            cursor = next_cursor
        if not rows:
            raise RuntimeError(f"No OHLCV data fetched for {symbol} {timeframe}")
        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    def save_ohlcv(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Path:
        path = settings.data_dir / f"{symbol_to_filename(symbol)}_{timeframe}.csv"
        df.to_csv(path, index=False)
        return path

    def download_symbol_bundle(self, symbol: str, timeframes: list[str], months: int) -> dict[str, Path]:
        saved: dict[str, Path] = {}
        for timeframe in timeframes:
            df = self.fetch_ohlcv_history(symbol, timeframe, months)
            saved[timeframe] = self.save_ohlcv(df, symbol, timeframe)
            print(f"{symbol} {timeframe} saved to {saved[timeframe]}")
        return saved


def load_local_ohlcv(symbol: str, timeframe: str) -> pd.DataFrame:
    path = settings.data_dir / f"{symbol_to_filename(symbol)}_{timeframe}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing local data: {path}")
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df
