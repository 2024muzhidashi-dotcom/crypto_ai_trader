from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import settings
from .explain import build_exit_reason
from .utils import calculate_max_drawdown, save_json


@dataclass
class Position:
    direction: str
    entry_time: pd.Timestamp
    entry_price: float
    initial_stop_loss_price: float
    stop_loss_price: float
    take_profit_price: float
    take_profit_price_2: float
    original_take_profit_price_1: float
    original_take_profit_price_2: float
    breakeven_trigger_price: float
    support_level: float
    resistance_level: float
    explanation: dict[str, Any]
    bars_held: int = 0
    partial_taken: bool = False
    breakeven_armed: bool = False
    size_fraction: float = 1.0


class Backtester:
    def __init__(
        self,
        fee_bps: float = settings.default_fee_bps,
        slippage_bps: float = settings.default_slippage_bps,
        initial_capital: float = 1000.0,
        position_pct: float = settings.default_position_pct,
        leverage: float = settings.default_leverage,
    ) -> None:
        self.fee_bps = fee_bps
        self.slippage_bps = slippage_bps
        self.initial_capital = initial_capital
        self.position_pct = position_pct
        self.leverage = leverage

    def run(self, df: pd.DataFrame, max_holding_bars: int = 240) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        equity = self.initial_capital
        position: Position | None = None
        trades: list[dict[str, Any]] = []
        equity_curve: list[dict[str, Any]] = []

        for _, row in df.iterrows():
            if position is not None:
                position.bars_held += 1
                exit_event = self._check_exit(position, row, max_holding_bars)
                if exit_event is not None:
                    trade = self._close_position(position, row, exit_event, equity)
                    equity += trade["pnl_amount"]
                    trades.append(trade)
                    if exit_event["reason"] == "take_profit_1" and not position.partial_taken:
                        position.partial_taken = True
                        position.size_fraction = 0.5
                        position.take_profit_price = position.take_profit_price_2
                        position.stop_loss_price = position.entry_price
                        position.breakeven_armed = True
                    else:
                        position = None

            if position is None and row["signal"] in {"long", "short"}:
                position = Position(
                    direction=row["signal"],
                    entry_time=row["timestamp"],
                    entry_price=float(row.get("planned_entry_price", row["close"])),
                    initial_stop_loss_price=float(row["stop_loss_price"]),
                    stop_loss_price=float(row["stop_loss_price"]),
                    take_profit_price=float(row["take_profit_price"]),
                    take_profit_price_2=float(row.get("take_profit_price_2", row["take_profit_price"])),
                    original_take_profit_price_1=float(row.get("take_profit_price_1", row["take_profit_price"])),
                    original_take_profit_price_2=float(row.get("take_profit_price_2", row["take_profit_price"])),
                    breakeven_trigger_price=float(row.get("breakeven_trigger_price", row["take_profit_price"])),
                    support_level=float(row["support_level"]),
                    resistance_level=float(row["resistance_level"]),
                    explanation=row["signal_explanation"],
                )

            floating = 0.0 if position is None else self._mark_to_market(position, float(row["close"]))
            equity_curve.append({"timestamp": row["timestamp"], "equity": equity + floating})

        trades_df = pd.DataFrame(trades)
        equity_df = pd.DataFrame(equity_curve)
        metrics = self._metrics(trades_df, equity_df)
        return trades_df, equity_df, metrics

    def _check_exit(self, position: Position, row: pd.Series, max_holding_bars: int) -> dict[str, Any] | None:
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])

        if position.direction == "long":
            if low <= position.stop_loss_price:
                return {"reason": "stop_loss", "exit_price": position.stop_loss_price}
            if not position.partial_taken and high >= position.take_profit_price:
                return {"reason": "take_profit_1", "exit_price": position.take_profit_price}
            if position.partial_taken and high >= position.take_profit_price:
                return {"reason": "take_profit_2", "exit_price": position.take_profit_price}
        else:
            if high >= position.stop_loss_price:
                return {"reason": "stop_loss", "exit_price": position.stop_loss_price}
            if not position.partial_taken and low <= position.take_profit_price:
                return {"reason": "take_profit_1", "exit_price": position.take_profit_price}
            if position.partial_taken and low <= position.take_profit_price:
                return {"reason": "take_profit_2", "exit_price": position.take_profit_price}

        if position.bars_held >= max_holding_bars:
            return {"reason": "time_exit", "exit_price": close}
        return None

    def _close_position(self, position: Position, row: pd.Series, exit_event: dict[str, Any], equity_before: float) -> dict[str, Any]:
        entry_price = position.entry_price
        exit_price = float(exit_event["exit_price"])
        margin = equity_before * self.position_pct * position.size_fraction
        notional = margin * self.leverage
        if position.direction == "long":
            gross = (exit_price - entry_price) * notional / entry_price
        else:
            gross = (entry_price - exit_price) * notional / entry_price
        fees = notional * 2 * self.fee_bps / 10000
        pnl = gross - fees
        pnl_pct_equity = pnl / equity_before if equity_before else 0.0
        pnl_pct_margin = pnl / margin if margin else 0.0
        explanation = dict(position.explanation)
        explanation["exit_reason"] = build_exit_reason(exit_event["reason"])
        return {
            "entry_time": position.entry_time,
            "exit_time": row["timestamp"],
            "direction": position.direction,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "initial_stop_loss_price": position.initial_stop_loss_price,
            "stop_loss_price": position.stop_loss_price,
            "take_profit_price": position.take_profit_price,
            "take_profit_price_1": position.original_take_profit_price_1,
            "take_profit_price_2": position.original_take_profit_price_2,
            "support_level": position.support_level,
            "resistance_level": position.resistance_level,
            "exit_type": exit_event["reason"],
            "pnl_amount": pnl,
            "pnl_pct_equity": pnl_pct_equity,
            "pnl_pct_margin": pnl_pct_margin,
            "margin_used": margin,
            "notional": notional,
            "entry_reason": position.explanation["entry_reason"],
            "exit_reason": explanation["exit_reason"],
            "signal_payload": explanation,
        }

    def _mark_to_market(self, position: Position, close_price: float) -> float:
        margin = self.initial_capital * self.position_pct * position.size_fraction
        notional = margin * self.leverage
        if position.direction == "long":
            return (close_price - position.entry_price) * notional / position.entry_price
        return (position.entry_price - close_price) * notional / position.entry_price

    def _metrics(self, trades_df: pd.DataFrame, equity_df: pd.DataFrame) -> dict[str, Any]:
        return {
            "total_trades": int(len(trades_df)),
            "win_rate": float((trades_df["pnl_amount"] > 0).mean()) if not trades_df.empty else 0.0,
            "final_equity": float(equity_df["equity"].iloc[-1]) if not equity_df.empty else self.initial_capital,
            "max_drawdown": calculate_max_drawdown(equity_df["equity"]) if not equity_df.empty else 0.0,
            "avg_pnl_pct_equity": float(trades_df["pnl_pct_equity"].mean()) if not trades_df.empty else 0.0,
        }


def simulate_ten_u_ladder(trades_df: pd.DataFrame, seed_capital: float = 10.0, refill_threshold: float = 2.0, target_capital: float = 100.0) -> dict[str, Any]:
    pod_balance = seed_capital
    reserve = 0.0
    external_topups = 0.0
    harvest_count = 0
    history: list[float] = [pod_balance]
    if trades_df.empty:
        return {
            "pod_balance": pod_balance,
            "reserve": reserve,
            "external_topups": external_topups,
            "harvest_count": harvest_count,
            "history": history,
        }
    for _, trade in trades_df.iterrows():
        pod_balance += pod_balance * float(trade.get("pnl_pct_equity", 0.0))
        if pod_balance >= target_capital:
            harvest_count += 1
            reserve += target_capital - seed_capital
            pod_balance = seed_capital
        if pod_balance < refill_threshold:
            topup = seed_capital - pod_balance
            external_topups += topup
            pod_balance = seed_capital
        history.append(pod_balance + reserve)
    return {
        "pod_balance": pod_balance,
        "reserve": reserve,
        "external_topups": external_topups,
        "harvest_count": harvest_count,
        "history": history,
        "total_value": pod_balance + reserve,
    }


def save_backtest_outputs(trades_df: pd.DataFrame, equity_df: pd.DataFrame, metrics: dict[str, Any], prefix: str) -> dict[str, Path]:
    trades_path = settings.backtests_dir / f"{prefix}_trades.csv"
    equity_path = settings.backtests_dir / f"{prefix}_equity.csv"
    metrics_path = settings.backtests_dir / f"{prefix}_metrics.json"
    serialized = trades_df.copy()
    if not serialized.empty and "signal_payload" in serialized.columns:
        serialized["signal_payload"] = serialized["signal_payload"].astype(str)
    serialized.to_csv(trades_path, index=False)
    equity_df.to_csv(equity_path, index=False)
    save_json(metrics, metrics_path)
    return {"trades": trades_path, "equity": equity_path, "metrics": metrics_path}
