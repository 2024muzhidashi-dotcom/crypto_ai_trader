from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class DailyTrendModel:
    def predict_full(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy().sort_values("timestamp").reset_index(drop=True)
        data["trend_prob_up"] = 0.5
        data["daily_trend_label"] = "neutral"
        return data


@dataclass
class ZoneModelArtifacts:
    report: dict


class ZoneModel:
    def fit_predict(self, df: pd.DataFrame) -> tuple[pd.DataFrame, ZoneModelArtifacts]:
        data = df.copy().sort_values("timestamp").reset_index(drop=True)
        data["zone_pred"] = 0
        data["zone_prob_long"] = 0.5
        data["zone_prob_short"] = 0.5
        return data, ZoneModelArtifacts(report={})
