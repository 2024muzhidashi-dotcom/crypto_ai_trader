from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from scipy.signal import argrelextrema
except ModuleNotFoundError:  # pragma: no cover
    argrelextrema = None


@dataclass
class SupportResistanceConfig:
    swing_order: int = 4
    reaction_lookahead_bars: int = 8
    min_reaction_atr: float = 0.85
    reaction_cluster_atr: float = 1.10
    zone_half_width_atr: float = 0.75
    min_reaction_touches: int = 3
    volume_bin_atr: float = 0.75
    volume_node_quantile: float = 0.82
    min_volume_node_score: float = 1.15
    merge_zone_atr: float = 0.85
    breakout_buffer_atr: float = 0.30
    breakout_confirm_bars: int = 2
    retest_tolerance_atr: float = 0.85
    retest_lookahead_bars: int = 32
    role_event_cooldown_bars: int = 24
    max_zones: int = 80


@dataclass
class LevelSnapshotConfig:
    lookback_days: int = 14
    recent_event_hours: int = 48
    nearest_each_side: int = 3
    max_display_zones: int = 8


def _fallback_local_extrema(values: np.ndarray, mode: str, order: int) -> np.ndarray:
    indices: list[int] = []
    for idx in range(order, len(values) - order):
        left = values[idx - order : idx]
        right = values[idx + 1 : idx + order + 1]
        if mode == "max":
            if np.all(values[idx] > left) and np.all(values[idx] >= right):
                indices.append(idx)
        else:
            if np.all(values[idx] < left) and np.all(values[idx] <= right):
                indices.append(idx)
    return np.asarray(indices, dtype=int)


def _local_extrema_indices(values: np.ndarray, order: int, mode: str) -> np.ndarray:
    if len(values) < order * 2 + 1:
        return np.asarray([], dtype=int)
    if argrelextrema is not None:
        comparator = np.greater_equal if mode == "max" else np.less_equal
        return argrelextrema(values, comparator, order=order)[0]
    return _fallback_local_extrema(values, mode, order)


def _safe_median_atr(data: pd.DataFrame) -> float:
    atr = pd.to_numeric(data.get("atr14", pd.Series(dtype=float)), errors="coerce").dropna()
    if atr.empty:
        high_low = (data["high"] - data["low"]).dropna()
        return float(high_low.median()) if not high_low.empty else 1.0
    value = float(atr.median())
    return value if np.isfinite(value) and value > 0 else 1.0


def _mean_timestamp(values: list[pd.Timestamp]) -> pd.Timestamp:
    if not values:
        return pd.NaT
    ns_values = np.asarray([value.value for value in values], dtype=np.int64)
    return pd.to_datetime(int(ns_values.mean()), utc=True)


def _reaction_points(data: pd.DataFrame, cfg: SupportResistanceConfig) -> pd.DataFrame:
    highs = data["high"].to_numpy(dtype=float)
    lows = data["low"].to_numpy(dtype=float)
    volumes = data["volume"].to_numpy(dtype=float)
    atr_values = data["atr14"].to_numpy(dtype=float) if "atr14" in data.columns else highs - lows
    volume_ma = pd.Series(volumes).rolling(20, min_periods=1).mean().to_numpy(dtype=float)

    points: list[dict[str, object]] = []
    peak_idx = _local_extrema_indices(highs, cfg.swing_order, "max")
    valley_idx = _local_extrema_indices(lows, cfg.swing_order, "min")

    for idx in peak_idx:
        if idx + 1 >= len(data):
            continue
        atr_now = atr_values[idx]
        if not np.isfinite(atr_now) or atr_now <= 0:
            continue
        end = min(idx + cfg.reaction_lookahead_bars + 1, len(data))
        reaction = highs[idx] - float(np.min(lows[idx + 1 : end]))
        reaction_atr = reaction / atr_now
        if reaction_atr < cfg.min_reaction_atr:
            continue
        volume_score = volumes[idx] / max(volume_ma[idx], 1e-9)
        points.append(
            {
                "timestamp": data.at[idx, "timestamp"],
                "price": float(highs[idx]),
                "side": "resistance",
                "reaction_atr": float(reaction_atr),
                "volume_score": float(volume_score) if np.isfinite(volume_score) else 1.0,
                "source": "reaction_high",
            }
        )

    for idx in valley_idx:
        if idx + 1 >= len(data):
            continue
        atr_now = atr_values[idx]
        if not np.isfinite(atr_now) or atr_now <= 0:
            continue
        end = min(idx + cfg.reaction_lookahead_bars + 1, len(data))
        reaction = float(np.max(highs[idx + 1 : end])) - lows[idx]
        reaction_atr = reaction / atr_now
        if reaction_atr < cfg.min_reaction_atr:
            continue
        volume_score = volumes[idx] / max(volume_ma[idx], 1e-9)
        points.append(
            {
                "timestamp": data.at[idx, "timestamp"],
                "price": float(lows[idx]),
                "side": "support",
                "reaction_atr": float(reaction_atr),
                "volume_score": float(volume_score) if np.isfinite(volume_score) else 1.0,
                "source": "reaction_low",
            }
        )

    return pd.DataFrame(points)


def _cluster_reaction_zones(
    data: pd.DataFrame,
    points: pd.DataFrame,
    cfg: SupportResistanceConfig,
) -> pd.DataFrame:
    if points.empty:
        return pd.DataFrame()

    median_atr = _safe_median_atr(data)
    cluster_width = median_atr * cfg.reaction_cluster_atr
    zone_half = median_atr * cfg.zone_half_width_atr
    points = points.sort_values("price").reset_index(drop=True)
    clusters: list[list[dict[str, object]]] = []
    current: list[dict[str, object]] = []

    for _, point in points.iterrows():
        item = point.to_dict()
        if not current:
            current.append(item)
            continue
        current_center = float(np.mean([float(x["price"]) for x in current]))
        candidate_prices = [float(x["price"]) for x in current] + [float(item["price"])]
        candidate_span = max(candidate_prices) - min(candidate_prices)
        if abs(float(item["price"]) - current_center) <= cluster_width and candidate_span <= cluster_width * 2.2:
            current.append(item)
        else:
            clusters.append(current)
            current = [item]
    if current:
        clusters.append(current)

    rows: list[dict[str, object]] = []
    for cluster in clusters:
        support_touches = sum(1 for item in cluster if item["side"] == "support")
        resistance_touches = sum(1 for item in cluster if item["side"] == "resistance")
        total_touches = support_touches + resistance_touches
        if total_touches < cfg.min_reaction_touches:
            continue

        prices = np.asarray([float(item["price"]) for item in cluster], dtype=float)
        reaction_score = float(np.mean([float(item["reaction_atr"]) for item in cluster]))
        volume_score = float(np.mean([float(item["volume_score"]) for item in cluster]))
        center = float(np.average(prices, weights=np.clip([float(item["reaction_atr"]) for item in cluster], 0.1, None)))
        timestamps = [pd.to_datetime(item["timestamp"], utc=True) for item in cluster]
        sorted_timestamps = sorted(timestamps)
        formation_time = sorted_timestamps[min(cfg.min_reaction_touches - 1, len(sorted_timestamps) - 1)]
        role = "support" if support_touches > resistance_touches else "resistance" if resistance_touches > support_touches else "both"
        distance_from_center = np.abs(prices - center)
        half_width = max(zone_half, float(np.percentile(distance_from_center, 75)) + zone_half * 0.20)
        half_width = min(half_width, cluster_width * 1.60)
        rows.append(
            {
                "center": center,
                "zone_low": float(center - half_width),
                "zone_high": float(center + half_width),
                "sources": "reaction",
                "support_touches": support_touches,
                "resistance_touches": resistance_touches,
                "touches": total_touches,
                "reaction_score": reaction_score,
                "volume_score": volume_score,
                "strength_score": total_touches * 1.2 + reaction_score + min(volume_score, 3.0),
                "first_seen": min(timestamps),
                "last_seen": max(timestamps),
                "formation_time": formation_time,
                "touch_mid_time": _mean_timestamp(timestamps),
                "initial_role": role,
            }
        )
    return pd.DataFrame(rows)


def _volume_profile_zones(data: pd.DataFrame, cfg: SupportResistanceConfig) -> pd.DataFrame:
    median_atr = _safe_median_atr(data)
    bin_width = max(median_atr * cfg.volume_bin_atr, 1e-9)
    price_low = float(data["low"].min())
    price_high = float(data["high"].max())
    if not np.isfinite(price_low) or not np.isfinite(price_high) or price_high <= price_low:
        return pd.DataFrame()

    bins = np.arange(price_low, price_high + bin_width, bin_width)
    if len(bins) < 4:
        return pd.DataFrame()

    typical_price = ((data["high"] + data["low"] + data["close"]) / 3.0).to_numpy(dtype=float)
    volume = data["volume"].to_numpy(dtype=float)
    hist, edges = np.histogram(typical_price, bins=bins, weights=volume)
    if not np.any(hist > 0):
        return pd.DataFrame()

    threshold = float(np.quantile(hist[hist > 0], cfg.volume_node_quantile))
    avg_volume = float(np.mean(hist[hist > 0]))
    rows: list[dict[str, object]] = []
    for idx, value in enumerate(hist):
        left_value = hist[idx - 1] if idx > 0 else 0.0
        right_value = hist[idx + 1] if idx + 1 < len(hist) else 0.0
        is_local_node = value >= left_value and value >= right_value
        volume_score = float(value / max(avg_volume, 1e-9))
        if value < threshold or not is_local_node or volume_score < cfg.min_volume_node_score:
            continue

        low = float(edges[idx])
        high = float(edges[idx + 1])
        center = (low + high) / 2.0
        rows.append(
            {
                "center": center,
                "zone_low": low,
                "zone_high": high,
                "sources": "volume_profile",
                "support_touches": 0,
                "resistance_touches": 0,
                "touches": 0,
                "reaction_score": 0.0,
                "volume_score": volume_score,
                "strength_score": 1.5 + min(volume_score, 5.0),
                "first_seen": data["timestamp"].iloc[0],
                "last_seen": data["timestamp"].iloc[-1],
                "formation_time": data["timestamp"].iloc[-1],
                "touch_mid_time": pd.NaT,
                "initial_role": "volume_zone",
            }
        )
    return pd.DataFrame(rows)


def _merge_zone_candidates(
    data: pd.DataFrame,
    zones: pd.DataFrame,
    cfg: SupportResistanceConfig,
) -> pd.DataFrame:
    if zones.empty:
        return zones.copy()

    median_atr = _safe_median_atr(data)
    merge_gap = median_atr * cfg.merge_zone_atr
    zones = zones.sort_values("center").reset_index(drop=True)
    groups: list[list[dict[str, object]]] = []
    current: list[dict[str, object]] = []

    for _, zone in zones.iterrows():
        item = zone.to_dict()
        if not current:
            current.append(item)
            continue
        current_center = float(np.average([float(x["center"]) for x in current], weights=[float(x["strength_score"]) for x in current]))
        current_width = max(float(x["zone_high"]) - float(x["zone_low"]) for x in current)
        next_width = float(item["zone_high"]) - float(item["zone_low"])
        max_allowed_gap = merge_gap + max(current_width, next_width) * 0.30
        if abs(float(item["center"]) - current_center) <= max_allowed_gap:
            current.append(item)
        else:
            groups.append(current)
            current = [item]
    if current:
        groups.append(current)

    rows: list[dict[str, object]] = []
    last_close = float(data["close"].iloc[-1])
    for group in groups:
        weights = np.asarray([max(float(item["strength_score"]), 0.1) for item in group], dtype=float)
        centers = np.asarray([float(item["center"]) for item in group], dtype=float)
        center = float(np.average(centers, weights=weights))
        zone_low = float(min(float(item["zone_low"]) for item in group))
        zone_high = float(max(float(item["zone_high"]) for item in group))
        sources = sorted({source for item in group for source in str(item["sources"]).split("+")})
        support_touches = int(sum(int(item["support_touches"]) for item in group))
        resistance_touches = int(sum(int(item["resistance_touches"]) for item in group))
        touches = support_touches + resistance_touches
        reaction_score = float(np.mean([float(item["reaction_score"]) for item in group]))
        volume_score = float(max(float(item["volume_score"]) for item in group))
        strength_score = float(sum(float(item["strength_score"]) for item in group))
        reaction_formation_times = [
            pd.to_datetime(item["formation_time"], utc=True)
            for item in group
            if "reaction" in str(item["sources"])
        ]
        all_formation_times = [pd.to_datetime(item["formation_time"], utc=True) for item in group]
        formation_time = min(reaction_formation_times) if reaction_formation_times else max(all_formation_times)
        if last_close > zone_high:
            current_role = "support"
        elif last_close < zone_low:
            current_role = "resistance"
        else:
            current_role = "active"

        rows.append(
            {
                "zone_id": f"SR{len(rows) + 1:03d}",
                "center": center,
                "zone_low": zone_low,
                "zone_high": zone_high,
                "width": zone_high - zone_low,
                "sources": "+".join(sources),
                "support_touches": support_touches,
                "resistance_touches": resistance_touches,
                "touches": touches,
                "reaction_score": reaction_score,
                "volume_score": volume_score,
                "strength_score": strength_score,
                "current_role": current_role,
                "first_seen": min(pd.to_datetime(item["first_seen"], utc=True) for item in group),
                "last_seen": max(pd.to_datetime(item["last_seen"], utc=True) for item in group),
                "formation_time": formation_time,
                "touch_mid_time": _mean_timestamp(
                    [pd.to_datetime(item["touch_mid_time"], utc=True) for item in group if pd.notna(item["touch_mid_time"])]
                ),
            }
        )

    result = pd.DataFrame(rows).sort_values("strength_score", ascending=False).head(cfg.max_zones).copy()
    return result.sort_values("center").reset_index(drop=True)


def _stable_above(data: pd.DataFrame, idx: int, zone_high: float, buffer: float, bars: int) -> bool:
    if idx - bars + 1 < 0:
        return False
    closes = data["close"].iloc[idx - bars + 1 : idx + 1].to_numpy(dtype=float)
    return bool(np.all(closes > zone_high + buffer))


def _stable_below(data: pd.DataFrame, idx: int, zone_low: float, buffer: float, bars: int) -> bool:
    if idx - bars + 1 < 0:
        return False
    closes = data["close"].iloc[idx - bars + 1 : idx + 1].to_numpy(dtype=float)
    return bool(np.all(closes < zone_low - buffer))


def _find_retest(
    data: pd.DataFrame,
    start_idx: int,
    zone_low: float,
    zone_high: float,
    tolerance: float,
    direction: str,
    max_bars: int,
) -> tuple[pd.Timestamp | pd.NaT, float | float, str]:
    end = min(start_idx + max_bars + 1, len(data))
    for idx in range(start_idx + 1, end):
        candle = data.iloc[idx]
        if direction == "up":
            touched = float(candle["low"]) <= zone_high + tolerance
            held = float(candle["close"]) >= zone_low
            if touched and held:
                return pd.to_datetime(candle["timestamp"], utc=True), float(candle["close"]), "retest_confirmed"
        else:
            touched = float(candle["high"]) >= zone_low - tolerance
            held = float(candle["close"]) <= zone_high
            if touched and held:
                return pd.to_datetime(candle["timestamp"], utc=True), float(candle["close"]), "retest_confirmed"
    return pd.NaT, np.nan, "break_confirmed"


def detect_role_reversal_events(
    df: pd.DataFrame,
    zones_df: pd.DataFrame,
    config: SupportResistanceConfig | None = None,
) -> pd.DataFrame:
    cfg = config or SupportResistanceConfig()
    if zones_df.empty:
        return pd.DataFrame()

    data = df.copy().sort_values("timestamp").reset_index(drop=True)
    median_atr = _safe_median_atr(data)
    buffer = median_atr * cfg.breakout_buffer_atr
    retest_tolerance = median_atr * cfg.retest_tolerance_atr
    events: list[dict[str, object]] = []

    for _, zone in zones_df.iterrows():
        zone_low = float(zone["zone_low"])
        zone_high = float(zone["zone_high"])
        formation_time = pd.to_datetime(zone.get("formation_time", zone.get("last_seen", data["timestamp"].iloc[0])), utc=True)
        start_candidates = data.index[data["timestamp"] > formation_time]
        if len(start_candidates) == 0:
            continue
        start_idx = max(int(start_candidates[0]), cfg.breakout_confirm_bars - 1)
        previous_stable_state = "inside"
        last_event_direction = ""
        last_event_idx = -10_000
        previous_close = float(data.at[start_idx - 1, "close"]) if start_idx > 0 else float(data.at[start_idx, "close"])
        if previous_close > zone_high + buffer:
            previous_stable_state = "above"
        elif previous_close < zone_low - buffer:
            previous_stable_state = "below"
        for idx in range(start_idx, len(data)):
            close = float(data.at[idx, "close"])
            if _stable_above(data, idx, zone_high, buffer, cfg.breakout_confirm_bars):
                stable_state = "above"
            elif _stable_below(data, idx, zone_low, buffer, cfg.breakout_confirm_bars):
                stable_state = "below"
            else:
                stable_state = "inside"

            if stable_state == "above" and previous_stable_state != "above" and last_event_direction != "up":
                if idx - last_event_idx < cfg.role_event_cooldown_bars:
                    previous_stable_state = stable_state
                    continue
                retest_time, retest_price, status = _find_retest(
                    data, idx, zone_low, zone_high, retest_tolerance, "up", cfg.retest_lookahead_bars
                )
                events.append(
                    {
                        "zone_id": zone["zone_id"],
                        "timestamp": data.at[idx, "timestamp"],
                        "event_type": "resistance_breakout_to_support",
                        "direction": "up",
                        "old_role": "resistance",
                        "new_role": "support",
                        "breakout_price": close,
                        "zone_low": zone_low,
                        "zone_high": zone_high,
                        "zone_center": float(zone["center"]),
                        "status": status,
                        "retest_time": retest_time,
                        "retest_price": retest_price,
                        "reason": "价格连续收在压力区上方，压力位进入支撑位观察；若随后回踩不破，则确认支撑压力转换。",
                    }
                )
                last_event_direction = "up"
                last_event_idx = idx

            if stable_state == "below" and previous_stable_state != "below" and last_event_direction != "down":
                if idx - last_event_idx < cfg.role_event_cooldown_bars:
                    previous_stable_state = stable_state
                    continue
                retest_time, retest_price, status = _find_retest(
                    data, idx, zone_low, zone_high, retest_tolerance, "down", cfg.retest_lookahead_bars
                )
                events.append(
                    {
                        "zone_id": zone["zone_id"],
                        "timestamp": data.at[idx, "timestamp"],
                        "event_type": "support_breakdown_to_resistance",
                        "direction": "down",
                        "old_role": "support",
                        "new_role": "resistance",
                        "breakout_price": close,
                        "zone_low": zone_low,
                        "zone_high": zone_high,
                        "zone_center": float(zone["center"]),
                        "status": status,
                        "retest_time": retest_time,
                        "retest_price": retest_price,
                        "reason": "价格连续收在支撑区下方，支撑位进入压力位观察；若随后回踩不过，则确认支撑压力转换。",
                    }
                )
                last_event_direction = "down"
                last_event_idx = idx

            previous_stable_state = stable_state

    if not events:
        return pd.DataFrame()
    return pd.DataFrame(events).sort_values("timestamp").reset_index(drop=True)


def detect_support_resistance_zones(
    df: pd.DataFrame,
    config: SupportResistanceConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = config or SupportResistanceConfig()
    data = df.copy().sort_values("timestamp").reset_index(drop=True)
    reaction_points = _reaction_points(data, cfg)
    reaction_zones = _cluster_reaction_zones(data, reaction_points, cfg)
    volume_zones = _volume_profile_zones(data, cfg)
    raw_zones = pd.concat([reaction_zones, volume_zones], ignore_index=True)
    zones = _merge_zone_candidates(data, raw_zones, cfg)
    events = detect_role_reversal_events(data, zones, cfg)
    return zones, events


def _role_cn(role: str) -> str:
    if role == "support":
        return "支撑"
    if role == "resistance":
        return "压力"
    if role == "active":
        return "当前价格所在区域"
    return role


def _sources_cn(sources: str) -> str:
    parts = set(str(sources).split("+"))
    labels: list[str] = []
    if "volume_profile" in parts:
        labels.append("密集成交区")
    if "reaction" in parts:
        labels.append("多次反弹/上冲失败区")
    return " + ".join(labels) if labels else str(sources)


def explain_zone(row: pd.Series, current_price: float) -> str:
    role = str(row["current_role"])
    source_text = _sources_cn(str(row["sources"]))
    relation = "价格正在区域内"
    if current_price > float(row["zone_high"]):
        relation = "当前价格已经站在该区域上方"
    elif current_price < float(row["zone_low"]):
        relation = "当前价格仍在该区域下方"

    touch_text = (
        f"共有 {int(row['touches'])} 次有效触碰，其中支撑反弹 {int(row['support_touches'])} 次、"
        f"压力回落 {int(row['resistance_touches'])} 次"
    )
    return (
        f"{source_text}；{touch_text}；强度分 {float(row['strength_score']):.1f}。"
        f"{relation}，所以当前判断为{_role_cn(role)}。"
    )


def explain_event(row: pd.Series) -> str:
    if str(row["direction"]) == "up":
        base = "价格有效上破原压力区，原压力位进入支撑位观察"
    else:
        base = "价格有效跌破原支撑区，原支撑位进入压力位观察"
    if str(row.get("status", "")) == "retest_confirmed":
        return f"{base}，并且后续回踩确认，转换更有效。"
    return f"{base}，但还没有看到明确回踩确认。"


def build_level_snapshot(
    df: pd.DataFrame,
    as_of: pd.Timestamp | None = None,
    sr_config: SupportResistanceConfig | None = None,
    snapshot_config: LevelSnapshotConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sr_cfg = sr_config or SupportResistanceConfig()
    snap_cfg = snapshot_config or LevelSnapshotConfig()
    data = df.copy().sort_values("timestamp").reset_index(drop=True)
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    if data.empty:
        return data, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    snapshot_time = pd.to_datetime(as_of, utc=True) if as_of is not None else data["timestamp"].iloc[-1]
    historical = data[data["timestamp"] <= snapshot_time].copy().reset_index(drop=True)
    if historical.empty:
        return historical, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    window_start = snapshot_time - pd.Timedelta(days=snap_cfg.lookback_days)
    context = historical[historical["timestamp"] >= window_start].copy().reset_index(drop=True)
    if len(context) < 80:
        context = historical.tail(80).copy().reset_index(drop=True)

    zones, events = detect_support_resistance_zones(context, sr_cfg)
    if zones.empty:
        return context, zones, events, pd.DataFrame()

    current_price = float(context["close"].iloc[-1])
    zones = zones.copy()
    zones["distance_to_price"] = np.where(
        current_price > zones["zone_high"],
        current_price - zones["zone_high"],
        np.where(current_price < zones["zone_low"], zones["zone_low"] - current_price, 0.0),
    )
    zones["distance_pct"] = zones["distance_to_price"] / max(current_price, 1e-9)

    active = zones[(zones["zone_low"] <= current_price) & (zones["zone_high"] >= current_price)].copy()
    supports = zones[zones["zone_high"] < current_price].sort_values(["distance_to_price", "strength_score"], ascending=[True, False])
    resistances = zones[zones["zone_low"] > current_price].sort_values(["distance_to_price", "strength_score"], ascending=[True, False])

    recent_cutoff = snapshot_time - pd.Timedelta(hours=snap_cfg.recent_event_hours)
    recent_events = events.copy()
    if not recent_events.empty:
        recent_events["timestamp"] = pd.to_datetime(recent_events["timestamp"], utc=True)
        recent_events = recent_events[recent_events["timestamp"] >= recent_cutoff].copy()

    event_zone_ids = set(recent_events["zone_id"].tolist()) if not recent_events.empty else set()
    event_zones = zones[zones["zone_id"].isin(event_zone_ids)]

    selected = pd.concat(
        [
            active,
            supports.head(snap_cfg.nearest_each_side),
            resistances.head(snap_cfg.nearest_each_side),
            event_zones,
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["zone_id"])

    if len(selected) > snap_cfg.max_display_zones:
        selected = selected.sort_values(["distance_to_price", "strength_score"], ascending=[True, False]).head(snap_cfg.max_display_zones)
    selected = selected.sort_values("center").reset_index(drop=True)

    selected["role_cn"] = selected["current_role"].map(_role_cn)
    selected["source_cn"] = selected["sources"].map(_sources_cn)
    selected["explanation"] = selected.apply(lambda row: explain_zone(row, current_price), axis=1)

    if not recent_events.empty:
        recent_events = recent_events[recent_events["zone_id"].isin(set(selected["zone_id"]))].copy()
        recent_events["explanation"] = recent_events.apply(explain_event, axis=1)

    return context, selected, recent_events, zones


def attach_nearest_levels(df: pd.DataFrame, zones_df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy().sort_values("timestamp").reset_index(drop=True)
    if zones_df.empty:
        data["nearest_support"] = np.nan
        data["nearest_resistance"] = np.nan
        return data

    supports = zones_df[zones_df["current_role"].isin(["support", "active"])].copy()
    resistances = zones_df[zones_df["current_role"].isin(["resistance", "active"])].copy()
    support_centers = supports["center"].to_numpy(dtype=float)
    resistance_centers = resistances["center"].to_numpy(dtype=float)

    nearest_support: list[float] = []
    nearest_resistance: list[float] = []
    for close in data["close"].to_numpy(dtype=float):
        below = support_centers[support_centers <= close]
        above = resistance_centers[resistance_centers >= close]
        nearest_support.append(float(below.max()) if len(below) else np.nan)
        nearest_resistance.append(float(above.min()) if len(above) else np.nan)
    data["nearest_support"] = nearest_support
    data["nearest_resistance"] = nearest_resistance
    return data


def build_levels(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    zones, _ = detect_support_resistance_zones(df)
    return zones


def merge_confluence_flags(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy()
