from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from scipy.signal import argrelextrema
except ModuleNotFoundError:  # pragma: no cover
    argrelextrema = None


@dataclass
class TriangleDetectorConfig:
    lookback_bars: int = 100
    extrema_order: int = 2
    min_peak_touches: int = 3
    min_valley_touches: int = 3
    max_regression_touches: int = 5
    touch_atr_pct: float = 0.35
    breakout_buffer_atr: float = 0.10
    pullback_tolerance_atr: float = 0.50
    max_pullback_bars: int = 12
    squeeze_ratio: float = 0.90
    volume_breakout_ratio: float = 1.5
    near_zero_slope: float = 0.00040
    max_touch_spacing_bars: int = 60
    breakout_body_atr: float = 0.08


@dataclass
class StructureScanConfig:
    lookback_bars: int = 48
    extrema_order: int = 3
    min_peak_touches: int = 3
    min_valley_touches: int = 3
    max_regression_touches: int = 12
    touch_atr_pct: float = 0.70
    zone_atr_pct: float = 0.95
    max_touch_spacing_bars: int = 48
    max_channel_width_atr: float = 12.0
    min_channel_width_atr: float = 0.6
    contraction_ratio: float = 1.20
    parallel_tolerance_atr: float = 2.4
    max_error_atr: float = 0.95
    recent_touch_window: int = 12
    max_window_atr_pct: float = 0.010
    max_close_std_pct: float = 0.018
    late_atr_ratio_max: float = 1.05
    max_prestructure_drift_pct: float = 0.02
    pivot_cluster_bars: int = 6
    min_pivot_prominence_atr: float = 0.35
    min_structure_bars: int = 18
    max_structure_bars: int = 56
    max_channel_width_pct: float = 0.018
    min_in_band_ratio: float = 0.78
    min_close_between_ratio: float = 0.84
    min_alternations: int = 5
    max_follow_bars: int = 96
    post_breakout_bars: int = 24
    breakout_buffer_atr: float = 0.18
    breakout_body_atr: float = 0.16
    breakout_body_outside_ratio: float = 0.25
    breakout_hold_bars: int = 3


def initialize_triangle_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    data = df.copy()
    numeric_nan = [
        "triangle_upper",
        "triangle_lower",
        "triangle_mid",
        "triangle_gap_start",
        "triangle_gap_end",
        "triangle_error",
        "triangle_tp1",
        "triangle_tp2",
        "triangle_stop",
        "triangle_recent_swing_high",
        "triangle_recent_swing_low",
        "triangle_touch_level",
        "triangle_pullback_anchor",
    ]
    numeric_zero = [
        "triangle_peak_touches",
        "triangle_valley_touches",
        "triangle_context",
        "triangle_breakout_long",
        "triangle_breakout_short",
        "triangle_pullback_long",
        "triangle_pullback_short",
        "triangle_signal_long",
        "triangle_signal_short",
    ]
    text_empty = [
        "triangle_state",
        "triangle_shape",
        "triangle_note",
    ]
    for column in numeric_nan:
        data[f"{prefix}_{column}"] = np.nan
    for column in numeric_zero:
        data[f"{prefix}_{column}"] = 0
    for column in text_empty:
        data[f"{prefix}_{column}"] = ""
    return data


def _fallback_local_extrema(values: np.ndarray, comparator: str, order: int) -> np.ndarray:
    indices: list[int] = []
    for idx in range(order, len(values) - order):
        left = values[idx - order : idx]
        right = values[idx + 1 : idx + order + 1]
        if comparator == "max":
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


def _fit_line(x_vals: np.ndarray, y_vals: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(x_vals.astype(float), y_vals.astype(float), 1)
    return float(slope), float(intercept)


def _line_at(slope: float, intercept: float, x_val: np.ndarray | float) -> np.ndarray | float:
    return slope * x_val + intercept


def _compress_extrema_points(
    indices: np.ndarray,
    values: np.ndarray,
    mode: str,
    min_spacing_bars: int,
    min_prominence: float,
) -> np.ndarray:
    if len(indices) == 0:
        return indices

    kept: list[int] = []
    cluster: list[int] = [int(indices[0])]

    def _flush(current_cluster: list[int]) -> None:
        if not current_cluster:
            return
        if mode == "max":
            best = max(current_cluster, key=lambda idx: values[idx])
            extreme = max(values[idx] for idx in current_cluster)
            base = min(values[idx] for idx in current_cluster)
        else:
            best = min(current_cluster, key=lambda idx: values[idx])
            extreme = min(values[idx] for idx in current_cluster)
            base = max(values[idx] for idx in current_cluster)
        if abs(extreme - base) >= min_prominence or len(current_cluster) == 1:
            kept.append(best)

    for idx in indices[1:]:
        idx = int(idx)
        if idx - cluster[-1] <= min_spacing_bars:
            cluster.append(idx)
        else:
            _flush(cluster)
            cluster = [idx]
    _flush(cluster)
    return np.asarray(sorted(set(kept)), dtype=int)


def _recent_horizontal_targets(window: pd.DataFrame, high_col: str, low_col: str, current_close: float) -> tuple[float, float]:
    swing_highs = window.loc[window[high_col] == window[high_col].rolling(3, center=True).max(), high_col].dropna()
    swing_lows = window.loc[window[low_col] == window[low_col].rolling(3, center=True).min(), low_col].dropna()
    highs_above = sorted([float(value) for value in swing_highs if value > current_close])
    lows_below = sorted([float(value) for value in swing_lows if value < current_close], reverse=True)
    next_high = highs_above[0] if highs_above else float(window[high_col].max())
    next_low = lows_below[0] if lows_below else float(window[low_col].min())
    return next_high, next_low


def _candle_stall(open_: float, high: float, low: float, close: float, direction: str) -> bool:
    candle_range = max(high - low, 1e-9)
    body = abs(close - open_)
    upper_wick = high - max(open_, close)
    lower_wick = min(open_, close) - low
    body_small = body / candle_range <= 0.45
    if direction == "long":
        return lower_wick / candle_range >= 0.30 or (close >= open_ and body_small)
    return upper_wick / candle_range >= 0.30 or (close <= open_ and body_small)


def _touch_count_near_line(
    x_idx: np.ndarray,
    y_vals: np.ndarray,
    slope: float,
    intercept: float,
    tolerance: float,
) -> int:
    if len(x_idx) == 0:
        return 0
    line_vals = _line_at(slope, intercept, x_idx)
    return int(np.sum(np.abs(y_vals - line_vals) <= tolerance))


def _recent_touch_ready(
    x_idx: np.ndarray,
    y_vals: np.ndarray,
    slope: float,
    intercept: float,
    tolerance: float,
    min_touches: int,
    max_touch_spacing_bars: int,
) -> bool:
    if len(x_idx) < min_touches:
        return False
    recent_x = x_idx[-min_touches:]
    recent_y = y_vals[-min_touches:]
    recent_line = _line_at(slope, intercept, recent_x)
    near_line = np.abs(recent_y - recent_line) <= tolerance
    spaced_tightly = (recent_x[-1] - recent_x[0]) <= max_touch_spacing_bars
    return bool(np.all(near_line) and spaced_tightly)


def _has_recent_zone_touch(
    x_idx: np.ndarray,
    y_vals: np.ndarray,
    slope: float,
    intercept: float,
    tolerance: float,
    window_end: int,
    recent_touch_window: int,
) -> bool:
    if len(x_idx) == 0:
        return False
    recent_mask = x_idx >= max(window_end - recent_touch_window, 0)
    if not np.any(recent_mask):
        return False
    recent_x = x_idx[recent_mask]
    recent_y = y_vals[recent_mask]
    recent_line = _line_at(slope, intercept, recent_x)
    return bool(np.any(np.abs(recent_y - recent_line) <= tolerance))


def _alternation_count(peak_idx: np.ndarray, valley_idx: np.ndarray) -> int:
    tagged = [(int(i), "peak") for i in peak_idx] + [(int(i), "valley") for i in valley_idx]
    if not tagged:
        return 0
    tagged.sort(key=lambda item: item[0])
    count = 0
    prev = tagged[0][1]
    for _, label in tagged[1:]:
        if label != prev:
            count += 1
            prev = label
    return count


def _first_channel_escape(
    window: pd.DataFrame,
    start_idx: int,
    upper_slope: float,
    upper_intercept: float,
    lower_slope: float,
    lower_intercept: float,
    zone_half_width: float,
) -> int | None:
    closes = window["close"].to_numpy(dtype=float)
    opens = window["open"].to_numpy(dtype=float)
    highs = window["high"].to_numpy(dtype=float)
    lows = window["low"].to_numpy(dtype=float)

    for idx in range(max(start_idx + 6, 2), len(window)):
        upper_now = float(_line_at(upper_slope, upper_intercept, idx))
        lower_now = float(_line_at(lower_slope, lower_intercept, idx))
        upper_prev = float(_line_at(upper_slope, upper_intercept, idx - 1))
        lower_prev = float(_line_at(lower_slope, lower_intercept, idx - 1))

        bullish_break = (
            closes[idx] > upper_now + zone_half_width * 0.35
            and closes[idx] > opens[idx]
            and closes[idx - 1] <= upper_prev + zone_half_width * 0.35
        )
        bearish_break = (
            closes[idx] < lower_now - zone_half_width * 0.35
            and closes[idx] < opens[idx]
            and closes[idx - 1] >= lower_prev - zone_half_width * 0.35
        )

        wick_expansion = (
            highs[idx] > upper_now + zone_half_width * 0.75
            or lows[idx] < lower_now - zone_half_width * 0.75
        )

        if bullish_break or bearish_break or wick_expansion:
            return idx
    return None


def detect_triangle_reversals(
    df: pd.DataFrame,
    prefix: str,
    config: TriangleDetectorConfig | None = None,
) -> pd.DataFrame:
    cfg = config or TriangleDetectorConfig()
    data = initialize_triangle_columns(df.copy(), prefix)
    close_col = f"{prefix}_close"
    open_col = f"{prefix}_open"
    high_col = f"{prefix}_high"
    low_col = f"{prefix}_low"
    atr_col = f"{prefix}_atr14"
    volume_col = f"{prefix}_volume"
    ema21_col = f"{prefix}_ema20"
    ema55_col = f"{prefix}_ema50"
    volume_ma = data[volume_col].rolling(20).mean()

    state = "wait_breakout"
    breakout_direction: str | None = None
    breakout_idx: int | None = None
    breakout_line = np.nan
    breakout_mid = np.nan
    breakout_stop = np.nan
    breakout_tp1 = np.nan
    breakout_tp2 = np.nan
    breakout_pullback_anchor = np.nan

    for idx in range(cfg.lookback_bars, len(data)):
        window = data.iloc[idx - cfg.lookback_bars : idx].copy().reset_index(drop=True)
        highs = window[high_col].to_numpy(dtype=float)
        lows = window[low_col].to_numpy(dtype=float)
        atr_now = float(data.at[idx, atr_col]) if pd.notna(data.at[idx, atr_col]) else np.nan
        if not np.isfinite(atr_now) or atr_now <= 0:
            state = "wait_breakout"
            breakout_direction = None
            breakout_idx = None
            continue

        peak_idx = _local_extrema_indices(highs, cfg.extrema_order, "max")
        valley_idx = _local_extrema_indices(lows, cfg.extrema_order, "min")
        if len(peak_idx) < cfg.min_peak_touches or len(valley_idx) < cfg.min_valley_touches:
            state = "wait_breakout"
            breakout_direction = None
            breakout_idx = None
            continue

        peak_idx = peak_idx[-cfg.max_regression_touches :]
        valley_idx = valley_idx[-cfg.max_regression_touches :]
        peak_y = highs[peak_idx]
        valley_y = lows[valley_idx]
        upper_slope, upper_intercept = _fit_line(peak_idx, peak_y)
        lower_slope, lower_intercept = _fit_line(valley_idx, valley_y)

        touch_tol = atr_now * cfg.touch_atr_pct
        peak_touches = _touch_count_near_line(peak_idx, peak_y, upper_slope, upper_intercept, touch_tol)
        valley_touches = _touch_count_near_line(valley_idx, valley_y, lower_slope, lower_intercept, touch_tol)

        x0 = 0.0
        x1 = float(cfg.lookback_bars - 1)
        upper_start = float(_line_at(upper_slope, upper_intercept, x0))
        lower_start = float(_line_at(lower_slope, lower_intercept, x0))
        upper_end = float(_line_at(upper_slope, upper_intercept, x1))
        lower_end = float(_line_at(lower_slope, lower_intercept, x1))
        gap_start = upper_start - lower_start
        gap_end = upper_end - lower_end
        if gap_start <= 0 or gap_end <= 0:
            state = "wait_breakout"
            breakout_direction = None
            breakout_idx = None
            continue

        quarter = max(cfg.lookback_bars // 4, 5)
        atr_early = float(window[atr_col].iloc[: cfg.lookback_bars - quarter].mean())
        atr_late = float(window[atr_col].iloc[-quarter:].mean())
        squeeze_ok = np.isfinite(atr_early) and np.isfinite(atr_late) and atr_late <= atr_early * cfg.squeeze_ratio

        upper_norm = upper_slope / max(abs(upper_end), 1.0)
        lower_norm = lower_slope / max(abs(lower_end), 1.0)
        slope_ok = upper_norm < cfg.near_zero_slope and lower_norm > -cfg.near_zero_slope
        convergence_ok = gap_end < gap_start * 0.95
        peak_line_vals = _line_at(upper_slope, upper_intercept, peak_idx)
        valley_line_vals = _line_at(lower_slope, lower_intercept, valley_idx)
        error = float((np.mean(np.abs(peak_y - peak_line_vals)) + np.mean(np.abs(valley_y - valley_line_vals))) / 2.0)
        recent_peak_ready = _recent_touch_ready(
            peak_idx,
            peak_y,
            upper_slope,
            upper_intercept,
            touch_tol,
            cfg.min_peak_touches,
            cfg.max_touch_spacing_bars,
        )
        recent_valley_ready = _recent_touch_ready(
            valley_idx,
            valley_y,
            lower_slope,
            lower_intercept,
            touch_tol,
            cfg.min_valley_touches,
            cfg.max_touch_spacing_bars,
        )
        context_ok = (
            peak_touches >= cfg.min_peak_touches
            and valley_touches >= cfg.min_valley_touches
            and slope_ok
            and convergence_ok
            and squeeze_ok
            and error <= atr_now * 0.55
            and recent_peak_ready
            and recent_valley_ready
        )

        current_upper = upper_end
        current_lower = lower_end
        current_mid = (current_upper + current_lower) / 2.0
        current_close = float(data.at[idx, close_col])
        current_open = float(data.at[idx, open_col])
        current_high = float(data.at[idx, high_col])
        current_low = float(data.at[idx, low_col])
        current_volume = float(data.at[idx, volume_col]) if pd.notna(data.at[idx, volume_col]) else 0.0
        avg_volume = float(volume_ma.iloc[idx]) if pd.notna(volume_ma.iloc[idx]) else np.nan
        breakout_buffer = atr_now * cfg.breakout_buffer_atr
        pullback_tolerance = atr_now * cfg.pullback_tolerance_atr

        data.at[idx, f"{prefix}_triangle_upper"] = current_upper
        data.at[idx, f"{prefix}_triangle_lower"] = current_lower
        data.at[idx, f"{prefix}_triangle_mid"] = current_mid
        data.at[idx, f"{prefix}_triangle_gap_start"] = gap_start
        data.at[idx, f"{prefix}_triangle_gap_end"] = gap_end
        data.at[idx, f"{prefix}_triangle_error"] = error
        data.at[idx, f"{prefix}_triangle_peak_touches"] = peak_touches
        data.at[idx, f"{prefix}_triangle_valley_touches"] = valley_touches
        data.at[idx, f"{prefix}_triangle_context"] = int(context_ok)
        data.at[idx, f"{prefix}_triangle_shape"] = "contracting_triangle" if context_ok else ""

        recent_high, recent_low = _recent_horizontal_targets(window, high_col, low_col, current_close)
        data.at[idx, f"{prefix}_triangle_recent_swing_high"] = recent_high
        data.at[idx, f"{prefix}_triangle_recent_swing_low"] = recent_low

        breakout_up = (
            context_ok
            and current_close > current_upper + breakout_buffer
            and min(current_open, current_close) > current_upper
            and abs(current_close - current_open) >= atr_now * cfg.breakout_body_atr
            and np.isfinite(avg_volume)
            and current_volume >= avg_volume * cfg.volume_breakout_ratio
        )
        breakout_down = (
            context_ok
            and current_close < current_lower - breakout_buffer
            and max(current_open, current_close) < current_lower
            and abs(current_close - current_open) >= atr_now * cfg.breakout_body_atr
            and np.isfinite(avg_volume)
            and current_volume >= avg_volume * cfg.volume_breakout_ratio
        )

        if state == "wait_breakout":
            if breakout_up:
                state = "wait_pullback"
                breakout_direction = "long"
                breakout_idx = idx
                breakout_line = current_upper
                breakout_mid = current_mid
                breakout_stop = current_mid - atr_now * 0.10
                breakout_tp1 = recent_high
                pole_height = max(current_upper - float(window[low_col].min()), atr_now)
                breakout_tp2 = breakout_tp1 + pole_height
                breakout_pullback_anchor = max(current_upper, float(window[high_col].iloc[peak_idx[-1]]))
                data.at[idx, f"{prefix}_triangle_breakout_long"] = 1
                data.at[idx, f"{prefix}_triangle_state"] = "wait_pullback_long"
                continue
            if breakout_down:
                state = "wait_pullback"
                breakout_direction = "short"
                breakout_idx = idx
                breakout_line = current_lower
                breakout_mid = current_mid
                breakout_stop = current_mid + atr_now * 0.10
                breakout_tp1 = recent_low
                pole_height = max(float(window[high_col].max()) - current_lower, atr_now)
                breakout_tp2 = breakout_tp1 - pole_height
                breakout_pullback_anchor = min(current_lower, float(window[low_col].iloc[valley_idx[-1]]))
                data.at[idx, f"{prefix}_triangle_breakout_short"] = 1
                data.at[idx, f"{prefix}_triangle_state"] = "wait_pullback_short"
                continue

        if state == "wait_pullback" and breakout_idx is not None and breakout_direction is not None:
            if idx - breakout_idx > cfg.max_pullback_bars:
                state = "wait_breakout"
                breakout_direction = None
                breakout_idx = None
                continue

            data.at[idx, f"{prefix}_triangle_pullback_anchor"] = breakout_pullback_anchor
            data.at[idx, f"{prefix}_triangle_stop"] = breakout_stop
            data.at[idx, f"{prefix}_triangle_tp1"] = breakout_tp1
            data.at[idx, f"{prefix}_triangle_tp2"] = breakout_tp2

            if breakout_direction == "long":
                pullback_zone = current_low <= breakout_pullback_anchor + pullback_tolerance
                ema_support = current_low <= float(data.at[idx, ema21_col]) * 1.002 or current_low <= float(data.at[idx, ema55_col]) * 1.002
                stall = _candle_stall(current_open, current_high, current_low, current_close, "long")
                recovery = current_close >= breakout_line and current_close > current_open
                if pullback_zone:
                    data.at[idx, f"{prefix}_triangle_pullback_long"] = 1
                if pullback_zone and ema_support and stall and recovery:
                    data.at[idx, f"{prefix}_triangle_signal_long"] = 1
                    data.at[idx, f"{prefix}_triangle_state"] = "signal_long"
                    data.at[idx, f"{prefix}_triangle_note"] = "breakout_retest_long"
                    state = "wait_breakout"
                    breakout_direction = None
                    breakout_idx = None
                continue

            pullback_zone = current_high >= breakout_pullback_anchor - pullback_tolerance
            ema_resistance = current_high >= float(data.at[idx, ema21_col]) * 0.998 or current_high >= float(data.at[idx, ema55_col]) * 0.998
            stall = _candle_stall(current_open, current_high, current_low, current_close, "short")
            rejection = current_close <= breakout_line and current_close < current_open
            if pullback_zone:
                data.at[idx, f"{prefix}_triangle_pullback_short"] = 1
            if pullback_zone and ema_resistance and stall and rejection:
                data.at[idx, f"{prefix}_triangle_signal_short"] = 1
                data.at[idx, f"{prefix}_triangle_state"] = "signal_short"
                data.at[idx, f"{prefix}_triangle_note"] = "breakdown_retest_short"
                state = "wait_breakout"
                breakout_direction = None
                breakout_idx = None

    return data


def scan_consolidation_structures(
    df_15m: pd.DataFrame,
    config: StructureScanConfig | None = None,
) -> pd.DataFrame:
    cfg = config or StructureScanConfig()
    m15 = df_15m.copy().sort_values("timestamp").reset_index(drop=True)
    data = m15.copy()
    candidates: list[dict[str, object]] = []

    for idx in range(cfg.lookback_bars, len(data)):
        window = data.iloc[idx - cfg.lookback_bars : idx].copy().reset_index(drop=True)
        atr_now = float(window["atr14"].iloc[-1]) if pd.notna(window["atr14"].iloc[-1]) else np.nan
        if not np.isfinite(atr_now) or atr_now <= 0:
            continue

        highs = window["high"].to_numpy(dtype=float)
        lows = window["low"].to_numpy(dtype=float)
        peak_idx = _local_extrema_indices(highs, cfg.extrema_order, "max")
        valley_idx = _local_extrema_indices(lows, cfg.extrema_order, "min")
        peak_idx = _compress_extrema_points(
            peak_idx,
            highs,
            "max",
            cfg.pivot_cluster_bars,
            atr_now * cfg.min_pivot_prominence_atr,
        )
        valley_idx = _compress_extrema_points(
            valley_idx,
            lows,
            "min",
            cfg.pivot_cluster_bars,
            atr_now * cfg.min_pivot_prominence_atr,
        )
        if len(peak_idx) < cfg.min_peak_touches or len(valley_idx) < cfg.min_valley_touches:
            continue

        peak_idx = peak_idx[-cfg.max_regression_touches :]
        valley_idx = valley_idx[-cfg.max_regression_touches :]
        peak_y = highs[peak_idx]
        valley_y = lows[valley_idx]
        upper_slope, upper_intercept = _fit_line(peak_idx, peak_y)
        lower_slope, lower_intercept = _fit_line(valley_idx, valley_y)
        touch_tol = atr_now * cfg.touch_atr_pct
        zone_half_width = atr_now * cfg.zone_atr_pct
        peak_touches = _touch_count_near_line(peak_idx, peak_y, upper_slope, upper_intercept, touch_tol)
        valley_touches = _touch_count_near_line(valley_idx, valley_y, lower_slope, lower_intercept, touch_tol)
        if peak_touches < cfg.min_peak_touches or valley_touches < cfg.min_valley_touches:
            continue

        peak_cluster_ready = _recent_touch_ready(
            peak_idx, peak_y, upper_slope, upper_intercept, touch_tol, cfg.min_peak_touches, cfg.max_touch_spacing_bars
        )
        valley_cluster_ready = _recent_touch_ready(
            valley_idx, valley_y, lower_slope, lower_intercept, touch_tol, cfg.min_valley_touches, cfg.max_touch_spacing_bars
        )
        recent_peak_touch = _has_recent_zone_touch(
            peak_idx,
            peak_y,
            upper_slope,
            upper_intercept,
            zone_half_width,
            cfg.lookback_bars - 1,
            cfg.recent_touch_window,
        )
        recent_valley_touch = _has_recent_zone_touch(
            valley_idx,
            valley_y,
            lower_slope,
            lower_intercept,
            zone_half_width,
            cfg.lookback_bars - 1,
            cfg.recent_touch_window,
        )
        if not ((peak_cluster_ready or recent_peak_touch) and (valley_cluster_ready or recent_valley_touch)):
            continue

        x0 = 0.0
        x1 = float(cfg.lookback_bars - 1)
        upper_start = float(_line_at(upper_slope, upper_intercept, x0))
        lower_start = float(_line_at(lower_slope, lower_intercept, x0))
        upper_end = float(_line_at(upper_slope, upper_intercept, x1))
        lower_end = float(_line_at(lower_slope, lower_intercept, x1))
        gap_start = upper_start - lower_start
        gap_end = upper_end - lower_end
        if gap_start <= 0 or gap_end <= 0:
            continue

        avg_close = float(window["close"].mean())
        if not np.isfinite(avg_close) or avg_close <= 0:
            continue
        width_ok = cfg.min_channel_width_atr * atr_now <= gap_end <= cfg.max_channel_width_atr * atr_now
        if not width_ok:
            continue
        if (gap_end / avg_close) > cfg.max_channel_width_pct:
            continue

        avg_atr = float(window["atr14"].mean())
        close_std_pct = float(window["close"].std(ddof=0) / avg_close) if len(window) > 1 else np.nan
        late_len = max(cfg.lookback_bars // 4, 8)
        early_len = max(cfg.lookback_bars // 2, 16)
        atr_early = float(window["atr14"].iloc[:early_len].mean())
        atr_late = float(window["atr14"].iloc[-late_len:].mean())
        atr_ratio = atr_late / max(atr_early, 1e-9) if np.isfinite(atr_early) and atr_early > 0 else np.nan
        if not np.isfinite(avg_atr) or (avg_atr / avg_close) > cfg.max_window_atr_pct:
            continue
        if not np.isfinite(close_std_pct) or close_std_pct > cfg.max_close_std_pct:
            continue
        if np.isfinite(atr_ratio) and atr_ratio > cfg.late_atr_ratio_max:
            continue

        converging = gap_end <= gap_start * cfg.contraction_ratio
        parallel = abs((upper_start - upper_end) - (lower_start - lower_end)) <= atr_now * cfg.parallel_tolerance_atr
        if not (converging or parallel):
            continue

        peak_line_vals = _line_at(upper_slope, upper_intercept, peak_idx)
        valley_line_vals = _line_at(lower_slope, lower_intercept, valley_idx)
        error = float((np.mean(np.abs(peak_y - peak_line_vals)) + np.mean(np.abs(valley_y - valley_line_vals))) / 2.0)
        shape = "converging_triangle" if converging and (upper_slope < 0 or lower_slope > 0) else "channel_box"
        if error > atr_now * cfg.max_error_atr:
            continue

        peak_line_vals_window = _line_at(upper_slope, upper_intercept, peak_idx)
        valley_line_vals_window = _line_at(lower_slope, lower_intercept, valley_idx)
        valid_peak_mask = np.abs(peak_y - peak_line_vals_window) <= zone_half_width
        valid_valley_mask = np.abs(valley_y - valley_line_vals_window) <= zone_half_width
        structure_peak_idx = peak_idx[valid_peak_mask]
        structure_valley_idx = valley_idx[valid_valley_mask]
        if len(structure_peak_idx) < cfg.min_peak_touches or len(structure_valley_idx) < cfg.min_valley_touches:
            continue

        structure_start_idx = int(min(structure_peak_idx.min(), structure_valley_idx.min()))
        structure_end_idx = int(max(structure_peak_idx.max(), structure_valley_idx.max()))
        escape_idx = _first_channel_escape(
            window,
            structure_start_idx,
            upper_slope,
            upper_intercept,
            lower_slope,
            lower_intercept,
            zone_half_width,
        )
        if escape_idx is not None and escape_idx > structure_start_idx + 8:
            structure_end_idx = min(structure_end_idx, escape_idx - 1)

        structure_peak_idx = structure_peak_idx[(structure_peak_idx >= structure_start_idx) & (structure_peak_idx <= structure_end_idx)]
        structure_valley_idx = structure_valley_idx[(structure_valley_idx >= structure_start_idx) & (structure_valley_idx <= structure_end_idx)]
        if len(structure_peak_idx) < cfg.min_peak_touches or len(structure_valley_idx) < cfg.min_valley_touches:
            continue

        structure_window = window.iloc[structure_start_idx : structure_end_idx + 1].copy()
        if len(structure_window) < cfg.min_structure_bars or len(structure_window) > cfg.max_structure_bars:
            continue

        structure_first_close = float(structure_window["close"].iloc[0])
        structure_last_close = float(structure_window["close"].iloc[-1])
        if structure_first_close <= 0:
            continue
        prestructure_drift_pct = abs(structure_last_close - structure_first_close) / structure_first_close
        if prestructure_drift_pct > cfg.max_prestructure_drift_pct:
            continue

        structure_x = np.arange(structure_start_idx, structure_end_idx + 1, dtype=float)
        upper_band = _line_at(upper_slope, upper_intercept, structure_x)
        lower_band = _line_at(lower_slope, lower_intercept, structure_x)
        highs_in_band = structure_window["high"].to_numpy(dtype=float) <= (upper_band + zone_half_width)
        lows_in_band = structure_window["low"].to_numpy(dtype=float) >= (lower_band - zone_half_width)
        closes_between = (
            (structure_window["close"].to_numpy(dtype=float) <= (upper_band + zone_half_width * 0.8))
            & (structure_window["close"].to_numpy(dtype=float) >= (lower_band - zone_half_width * 0.8))
        )
        in_band_ratio = float(np.mean(highs_in_band & lows_in_band))
        close_between_ratio = float(np.mean(closes_between))
        if in_band_ratio < cfg.min_in_band_ratio or close_between_ratio < cfg.min_close_between_ratio:
            continue

        alternations = _alternation_count(structure_peak_idx, structure_valley_idx)
        if alternations < cfg.min_alternations:
            continue

        early_third = max(len(structure_window) // 3, 1)
        late_third_start = len(structure_window) - early_third
        early_peaks = int(np.sum(structure_peak_idx <= structure_start_idx + early_third))
        late_peaks = int(np.sum(structure_peak_idx >= structure_start_idx + late_third_start))
        early_valleys = int(np.sum(structure_valley_idx <= structure_start_idx + early_third))
        late_valleys = int(np.sum(structure_valley_idx >= structure_start_idx + late_third_start))
        if min(early_peaks + early_valleys, late_peaks + late_valleys) < 2:
            continue

        clipped_start_idx = max(structure_start_idx - 1, 0)
        clipped_end_idx = min(structure_end_idx + 1, len(window) - 1)
        clipped_window_start = window["timestamp"].iloc[clipped_start_idx]
        clipped_window_end = window["timestamp"].iloc[clipped_end_idx]
        clipped_upper_start = float(_line_at(upper_slope, upper_intercept, clipped_start_idx))
        clipped_upper_end = float(_line_at(upper_slope, upper_intercept, clipped_end_idx))
        clipped_lower_start = float(_line_at(lower_slope, lower_intercept, clipped_start_idx))
        clipped_lower_end = float(_line_at(lower_slope, lower_intercept, clipped_end_idx))

        structure_peak_y = highs[structure_peak_idx]
        structure_valley_y = lows[structure_valley_idx]

        seed_peak_idx = int(structure_peak_idx[0])
        seed_valley_idx = int(structure_valley_idx[0])
        peak_touches = int(len(structure_peak_idx))
        valley_touches = int(len(structure_valley_idx))

        candidates.append(
            {
                "timestamp": data.at[idx, "timestamp"],
                "window_start": clipped_window_start,
                "window_end": clipped_window_end,
                "upper_slope": upper_slope,
                "upper_intercept": upper_intercept,
                "lower_slope": lower_slope,
                "lower_intercept": lower_intercept,
                "upper_start": clipped_upper_start,
                "upper_end": clipped_upper_end,
                "lower_start": clipped_lower_start,
                "lower_end": clipped_lower_end,
                "mid_start": (clipped_upper_start + clipped_lower_start) / 2.0,
                "mid_end": (clipped_upper_end + clipped_lower_end) / 2.0,
                "gap_start": gap_start,
                "gap_end": gap_end,
                "peak_touches": peak_touches,
                "valley_touches": valley_touches,
                "shape": shape,
                "error": error,
                "zone_half_width": zone_half_width,
                "in_band_ratio": in_band_ratio,
                "close_between_ratio": close_between_ratio,
                "alternations": alternations,
                "anchor_peak_time": str(window["timestamp"].iloc[seed_peak_idx]),
                "anchor_valley_time": str(window["timestamp"].iloc[seed_valley_idx]),
                "anchor_peak_price": float(window["high"].iloc[seed_peak_idx]),
                "anchor_valley_price": float(window["low"].iloc[seed_valley_idx]),
                "peak_touch_times": [str(window["timestamp"].iloc[i]) for i in structure_peak_idx],
                "valley_touch_times": [str(window["timestamp"].iloc[i]) for i in structure_valley_idx],
                "peak_touch_prices": [float(window["high"].iloc[i]) for i in structure_peak_idx],
                "valley_touch_prices": [float(window["low"].iloc[i]) for i in structure_valley_idx],
            }
        )

    if not candidates:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "window_start",
                "window_end",
                "upper_start",
                "upper_end",
                "lower_start",
                "lower_end",
                "mid_start",
                "mid_end",
                "gap_start",
                "gap_end",
                "peak_touches",
                "valley_touches",
                "shape",
                "error",
                "peak_touch_times",
                "valley_touch_times",
                "peak_touch_prices",
                "valley_touch_prices",
            ]
        )

    result = pd.DataFrame(candidates).sort_values("timestamp").reset_index(drop=True)
    return result


def merge_structure_candidates(
    candidates_df: pd.DataFrame,
    max_time_gap_bars: int = 8,
) -> pd.DataFrame:
    if candidates_df.empty:
        return candidates_df.copy()

    data = candidates_df.copy().sort_values("timestamp").reset_index(drop=True)
    for col in ["timestamp", "window_start", "window_end"]:
        data[col] = pd.to_datetime(data[col], utc=True)
    merged_rows: list[dict[str, object]] = []
    current = data.iloc[0].to_dict()

    def _row_score(row: dict[str, object]) -> tuple[float, float]:
        touches = float(row["peak_touches"]) + float(row["valley_touches"])
        error = float(row["error"])
        return touches, -error

    def _interval_minutes(start: pd.Timestamp, end: pd.Timestamp) -> float:
        return max((end - start).total_seconds() / 60.0, 1.0)

    def _is_same_region(left: dict[str, object], right: dict[str, object]) -> bool:
        left_start = pd.to_datetime(left["window_start"], utc=True)
        left_end = pd.to_datetime(left["window_end"], utc=True)
        right_start = pd.to_datetime(right["window_start"], utc=True)
        right_end = pd.to_datetime(right["window_end"], utc=True)

        gap_minutes = (right_start - left_end).total_seconds() / 60.0
        if gap_minutes > 15 * max_time_gap_bars:
            return False

        overlap_start = max(left_start, right_start)
        overlap_end = min(left_end, right_end)
        overlap_minutes = max((overlap_end - overlap_start).total_seconds() / 60.0, 0.0)
        base_minutes = min(_interval_minutes(left_start, left_end), _interval_minutes(right_start, right_end))
        overlap_ratio = overlap_minutes / base_minutes
        same_shape = str(left["shape"]) == str(right["shape"])
        return bool(overlap_ratio >= 0.78 and same_shape)

    for _, next_row in data.iloc[1:].iterrows():
        row = next_row.to_dict()
        current_time = pd.to_datetime(current["timestamp"], utc=True)
        next_time = pd.to_datetime(row["timestamp"], utc=True)

        if _is_same_region(current, row):
            keep = row if _row_score(row) > _row_score(current) else current
            merged_start = min(pd.to_datetime(current["window_start"], utc=True), pd.to_datetime(row["window_start"], utc=True))
            merged_end = max(pd.to_datetime(current["window_end"], utc=True), pd.to_datetime(row["window_end"], utc=True))
            current = dict(keep)
            current["window_start"] = merged_start
            current["window_end"] = merged_end
            current["timestamp"] = max(current_time, next_time)
            current["peak_touches"] = max(int(current["peak_touches"]), int(row["peak_touches"]))
            current["valley_touches"] = max(int(current["valley_touches"]), int(row["valley_touches"]))
            current["zone_half_width"] = max(float(current["zone_half_width"]), float(row["zone_half_width"]))
            continue

        merged_rows.append(current)
        current = row

    merged_rows.append(current)
    result = pd.DataFrame(merged_rows)
    for col in ["timestamp", "window_start", "window_end"]:
        result[col] = pd.to_datetime(result[col], utc=True)
    return result.sort_values("timestamp").reset_index(drop=True)


def _bars_from_time(start: pd.Timestamp, current: pd.Timestamp, bar_minutes: int = 15) -> float:
    return (current - start).total_seconds() / (bar_minutes * 60.0)


def _structure_line_at(row: pd.Series | dict[str, object], timestamp: pd.Timestamp, side: str) -> float:
    start_time = pd.to_datetime(row["window_start"], utc=True)
    end_time = pd.to_datetime(row["window_end"], utc=True)
    duration_bars = max(_bars_from_time(start_time, end_time), 1.0)
    x = _bars_from_time(start_time, timestamp)

    if side == "upper":
        start_value = float(row["upper_start"])
        end_value = float(row["upper_end"])
    else:
        start_value = float(row["lower_start"])
        end_value = float(row["lower_end"])

    slope_per_bar = (end_value - start_value) / duration_bars
    return start_value + slope_per_bar * x


def annotate_structure_breakouts(
    candles_df: pd.DataFrame,
    structures_df: pd.DataFrame,
    config: StructureScanConfig | None = None,
) -> pd.DataFrame:
    """Follow each mature channel until price effectively leaves it.

    This is intentionally a review/monitoring layer, not a trade entry rule yet.
    It records where the market first proves that the channel has changed state.
    """
    if structures_df.empty:
        return structures_df.copy()

    cfg = config or StructureScanConfig()
    candles = candles_df.copy().sort_values("timestamp").reset_index(drop=True)
    candles["timestamp"] = pd.to_datetime(candles["timestamp"], utc=True)
    rows: list[dict[str, object]] = []

    for _, item in structures_df.iterrows():
        row = item.to_dict()
        channel_end = pd.to_datetime(row["window_end"], utc=True)
        future = candles[candles["timestamp"] > channel_end].head(cfg.max_follow_bars)

        breakout_direction = "pending"
        breakout_time: pd.Timestamp | pd.NaT = pd.NaT
        breakout_price = np.nan
        breakout_line_price = np.nan
        breakout_upper_price = np.nan
        breakout_lower_price = np.nan
        breakout_body_atr = np.nan
        breakout_bars_after_channel = np.nan

        for future_idx, candle in future.iterrows():
            timestamp = pd.to_datetime(candle["timestamp"], utc=True)
            upper_now = _structure_line_at(row, timestamp, "upper")
            lower_now = _structure_line_at(row, timestamp, "lower")
            atr_now = float(candle.get("atr14", np.nan))
            if not np.isfinite(atr_now) or atr_now <= 0:
                continue

            open_price = float(candle["open"])
            close_price = float(candle["close"])
            body = abs(close_price - open_price)
            if body < atr_now * cfg.breakout_body_atr:
                continue

            buffer = atr_now * cfg.breakout_buffer_atr
            bullish_body_outside = close_price - max(open_price, upper_now)
            bearish_body_outside = min(open_price, lower_now) - close_price
            bullish_breakout = (
                close_price > upper_now + buffer
                and close_price > open_price
                and bullish_body_outside >= body * cfg.breakout_body_outside_ratio
            )
            bearish_breakout = (
                close_price < lower_now - buffer
                and close_price < open_price
                and bearish_body_outside >= body * cfg.breakout_body_outside_ratio
            )

            if bullish_breakout or bearish_breakout:
                if cfg.breakout_hold_bars > 0:
                    hold_end_idx = future_idx + cfg.breakout_hold_bars
                    if hold_end_idx >= len(candles):
                        continue
                    hold = candles.iloc[future_idx + 1 : hold_end_idx + 1]
                    if len(hold) < cfg.breakout_hold_bars:
                        continue

                    held_outside = True
                    confirm_candle = hold.iloc[-1]
                    confirm_time = pd.to_datetime(confirm_candle["timestamp"], utc=True)
                    for _, hold_candle in hold.iterrows():
                        hold_time = pd.to_datetime(hold_candle["timestamp"], utc=True)
                        hold_close = float(hold_candle["close"])
                        hold_upper = _structure_line_at(row, hold_time, "upper")
                        hold_lower = _structure_line_at(row, hold_time, "lower")
                        if bullish_breakout and hold_close <= hold_upper:
                            held_outside = False
                            break
                        if bearish_breakout and hold_close >= hold_lower:
                            held_outside = False
                            break
                    if not held_outside:
                        continue
                else:
                    confirm_candle = candle
                    confirm_time = timestamp
                    hold_end_idx = future_idx

                confirm_upper = _structure_line_at(row, confirm_time, "upper")
                confirm_lower = _structure_line_at(row, confirm_time, "lower")
                breakout_direction = "up" if bullish_breakout else "down"
                breakout_time = confirm_time
                breakout_price = float(confirm_candle["close"])
                breakout_line_price = confirm_upper if bullish_breakout else confirm_lower
                breakout_upper_price = confirm_upper
                breakout_lower_price = confirm_lower
                breakout_body_atr = body / atr_now
                breakout_bars_after_channel = hold_end_idx - int(future.index[0])
                break

        if pd.notna(breakout_time):
            breakout_position = candles.index[candles["timestamp"] == breakout_time]
            if len(breakout_position) > 0:
                follow_end_idx = min(int(breakout_position[0]) + cfg.post_breakout_bars, len(candles) - 1)
                follow_end = candles.at[follow_end_idx, "timestamp"]
            else:
                follow_end = breakout_time
        elif not future.empty:
            follow_end = future["timestamp"].iloc[-1]
        else:
            follow_end = channel_end

        row.update(
            {
                "follow_end": follow_end,
                "breakout_direction": breakout_direction,
                "breakout_time": breakout_time,
                "breakout_price": breakout_price,
                "breakout_line_price": breakout_line_price,
                "breakout_upper_price": breakout_upper_price,
                "breakout_lower_price": breakout_lower_price,
                "breakout_body_atr": breakout_body_atr,
                "breakout_bars_after_channel": breakout_bars_after_channel,
            }
        )
        rows.append(row)

    result = pd.DataFrame(rows)
    for col in ["timestamp", "window_start", "window_end", "follow_end", "breakout_time"]:
        if col in result.columns:
            result[col] = pd.to_datetime(result[col], utc=True, errors="coerce")
    return result.reset_index(drop=True)
