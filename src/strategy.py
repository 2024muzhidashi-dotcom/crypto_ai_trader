from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class StrategyConfig:
    leverage: float = 100.0
    position_pct: float = 0.05
    min_rr: float = 1.5


class CleanSlateStrategy:
    """
    Fresh-start strategy scaffold.

    This file intentionally contains no legacy trading logic.
    We only keep the signal schema so the rest of the project can
    continue to run while we rebuild the system from zero.
    """

    def __init__(self, config: StrategyConfig | None = None) -> None:
        self.config = config or StrategyConfig()

    def prepare_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy().reset_index(drop=True)
        data["signal"] = "hold"
        data["entry_reason"] = ""
        data["stop_loss_price"] = np.nan
        data["take_profit_price"] = np.nan
        data["take_profit_price_1"] = np.nan
        data["take_profit_price_2"] = np.nan
        data["breakeven_trigger_price"] = np.nan
        data["risk_reward_ratio"] = np.nan
        data["support_level"] = np.nan
        data["resistance_level"] = np.nan
        data["signal_explanation"] = None
        data["planned_entry_price"] = np.nan
        return data

    def build_placeholder_signal(
        self,
        timestamp: pd.Timestamp,
        direction: str,
        entry_price: float,
        stop_loss_price: float,
        take_profit_price_1: float,
        take_profit_price_2: float,
        entry_reason: str,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        reward = abs(take_profit_price_2 - entry_price)
        risk = abs(entry_price - stop_loss_price)
        rr = reward / risk if risk > 0 else 0.0
        explanation = {
            "timestamp": str(timestamp),
            "strategy_name": "clean_slate_placeholder",
            "entry_reason": entry_reason,
            "risk_reward_ratio": rr,
        }
        if extra:
            explanation.update(extra)
        return {
            "signal": direction,
            "planned_entry_price": entry_price,
            "entry_reason": entry_reason,
            "stop_loss_price": stop_loss_price,
            "take_profit_price": take_profit_price_1,
            "take_profit_price_1": take_profit_price_1,
            "take_profit_price_2": take_profit_price_2,
            "breakeven_trigger_price": take_profit_price_1,
            "risk_reward_ratio": rr,
            "support_level": np.nan,
            "resistance_level": np.nan,
            "signal_explanation": explanation,
        }
