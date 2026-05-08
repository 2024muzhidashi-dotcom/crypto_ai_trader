from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover
    def load_dotenv(*_args, **_kwargs):
        return False


ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


@dataclass
class Settings:
    root_dir: Path = ROOT_DIR
    data_dir: Path = ROOT_DIR / "data"
    model_dir: Path = ROOT_DIR / "models"
    logs_dir: Path = ROOT_DIR / "logs"
    backtests_dir: Path = ROOT_DIR / "backtests"
    dashboard_dir: Path = ROOT_DIR / "dashboard"
    exchange_id: str = os.getenv("DEFAULT_EXCHANGE", "okx")
    base_asset: str = os.getenv("DEFAULT_BASE_ASSET", "BTC")
    symbol: str = os.getenv("DEFAULT_SYMBOL", "BTC/USDT:USDT")
    timeframes: list[str] = field(
        default_factory=lambda: [x.strip() for x in os.getenv("DEFAULT_TIMEFRAMES", "1d,4h,1h,15m,5m,1m").split(",")]
    )
    lookback_months: int = int(os.getenv("LOOKBACK_MONTHS", "36"))
    default_fee_bps: float = float(os.getenv("DEFAULT_FEE_BPS", "10"))
    default_slippage_bps: float = float(os.getenv("DEFAULT_SLIPPAGE_BPS", "5"))
    default_leverage: float = float(os.getenv("DEFAULT_LEVERAGE", "20"))
    default_position_pct: float = float(os.getenv("DEFAULT_POSITION_PCT", "0.05"))
    enable_live_trading: bool = os.getenv("ENABLE_LIVE_TRADING", "false").lower() == "true"
    okx_api_key: str = os.getenv("OKX_API_KEY", "")
    okx_secret: str = os.getenv("OKX_SECRET", "")
    okx_password: str = os.getenv("OKX_PASSWORD", "")

    def ensure_dirs(self) -> None:
        for path in [self.data_dir, self.model_dir, self.logs_dir, self.backtests_dir, self.dashboard_dir]:
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
