from __future__ import annotations


def build_exit_reason(reason: str) -> str:
    mapping = {
        "stop_loss": "止损",
        "take_profit_1": "止盈一半",
        "take_profit_2": "二次止盈",
        "time_exit": "时间退出",
    }
    return mapping.get(reason, reason)
