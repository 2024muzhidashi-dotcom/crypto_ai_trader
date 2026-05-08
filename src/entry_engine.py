from __future__ import annotations

import pandas as pd


def build_entry_timing(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["entry_timing_long"] = 0
    data["entry_timing_short"] = 0
    data["m5_long_ready"] = 0
    data["m5_short_ready"] = 0
    data["m1_long_trigger"] = 0
    data["m1_short_trigger"] = 0
    return data
