from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from .config import settings


def clear_review_outputs() -> list[Path]:
    backtests_dir = settings.backtests_dir
    removed: list[Path] = []
    for path in backtests_dir.iterdir():
        if path.is_dir():
            shutil.rmtree(path)
            removed.append(path)
        else:
            path.unlink()
            removed.append(path)
    return removed


def _safe_name(timestamp: pd.Timestamp, direction: str, seq: int) -> str:
    return f"trade_{seq:02d}_{direction}_{timestamp.strftime('%Y%m%d_%H%M')}.html"


def _bars_from_time(start: pd.Timestamp, current: pd.Timestamp, bar_minutes: int = 15) -> float:
    return (current - start).total_seconds() / (bar_minutes * 60.0)


def _project_structure_line(row: pd.Series, timestamp: pd.Timestamp, side: str) -> float:
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


def export_triangle_review(
    signals_df: pd.DataFrame,
    symbol_label: str,
    review_name: str = "triangle_review",
) -> dict[str, Path]:
    import plotly.graph_objects as go

    review_dir = settings.backtests_dir / review_name
    if review_dir.exists():
        shutil.rmtree(review_dir)
    review_dir.mkdir(parents=True, exist_ok=True)

    signal_rows = signals_df[
        (signals_df["m15_triangle_signal_long"] == 1) | (signals_df["m15_triangle_signal_short"] == 1)
    ].copy()
    review_mode = "signal"
    if signal_rows.empty:
        signal_rows = signals_df[
            (signals_df["m15_triangle_breakout_long"] == 1)
            | (signals_df["m15_triangle_breakout_short"] == 1)
            | (signals_df["m15_triangle_pullback_long"] == 1)
            | (signals_df["m15_triangle_pullback_short"] == 1)
        ].copy()
        review_mode = "candidate"
    if signal_rows.empty:
        signal_rows = signals_df[signals_df["m15_triangle_context"] == 1].copy()
        review_mode = "context"
    if signal_rows.empty:
        index_path = review_dir / "index.html"
        index_path.write_text("<html><body><h1>No triangle signals</h1></body></html>", encoding="utf-8")
        trade_index_path = review_dir / "trade_index.csv"
        pd.DataFrame(columns=["trade_id", "timestamp", "direction", "file"]).to_csv(trade_index_path, index=False)
        return {"review_dir": review_dir, "index": index_path, "trade_index": trade_index_path}

    trade_index_rows: list[dict[str, str]] = []
    links: list[str] = []

    for seq, (_, signal) in enumerate(signal_rows.iterrows(), start=1):
        timestamp = pd.to_datetime(signal["timestamp"], utc=True)
        if int(signal.get("m15_triangle_signal_long", 0)) == 1 or int(signal.get("m15_triangle_pullback_long", 0)) == 1 or int(signal.get("m15_triangle_breakout_long", 0)) == 1:
            direction = "long"
        else:
            direction = "short"
        start = timestamp - pd.Timedelta(hours=24)
        end = timestamp + pd.Timedelta(hours=18)
        view = signals_df[(signals_df["timestamp"] >= start) & (signals_df["timestamp"] <= end)].copy()
        if view.empty:
            continue

        filename = _safe_name(timestamp, direction, seq)
        file_path = review_dir / filename

        fig = go.Figure()
        fig.add_trace(
            go.Candlestick(
                x=view["timestamp"],
                open=view["m15_open"],
                high=view["m15_high"],
                low=view["m15_low"],
                close=view["m15_close"],
                name="15m K",
            )
        )

        if "m15_ema20" in view.columns:
            fig.add_trace(go.Scatter(x=view["timestamp"], y=view["m15_ema20"], mode="lines", name="EMA21", line={"color": "#f97316"}))
        if "m15_ema50" in view.columns:
            fig.add_trace(go.Scatter(x=view["timestamp"], y=view["m15_ema50"], mode="lines", name="EMA55", line={"color": "#10b981"}))

        upper_mask = view["m15_triangle_upper"].notna()
        lower_mask = view["m15_triangle_lower"].notna()
        if upper_mask.any() and lower_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=view.loc[upper_mask, "timestamp"],
                    y=view.loc[upper_mask, "m15_triangle_upper"],
                    mode="lines",
                    name="箱体上沿/压力线",
                    line={"color": "#2563eb", "width": 2},
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=view.loc[lower_mask, "timestamp"],
                    y=view.loc[lower_mask, "m15_triangle_lower"],
                    mode="lines",
                    name="箱体下沿/支撑线",
                    line={"color": "#1d4ed8", "width": 2},
                    fill="tonexty",
                    fillcolor="rgba(59,130,246,0.10)",
                )
            )
        mid_mask = view["m15_triangle_mid"].notna()
        if mid_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=view.loc[mid_mask, "timestamp"],
                    y=view.loc[mid_mask, "m15_triangle_mid"],
                    mode="lines",
                    name="箱体中轨",
                    line={"color": "#7c3aed", "dash": "dot", "width": 1.5},
                )
            )

        fig.add_trace(
            go.Scatter(
                x=[timestamp],
                y=[float(signal["m15_close"])],
                mode="markers+text",
                marker={
                    "symbol": "triangle-up" if direction == "long" else "triangle-down",
                    "size": 13,
                    "color": "#16a34a" if direction == "long" else "#dc2626",
                },
                text=["Entry"],
                textposition="top center",
                name="Entry",
            )
        )

        stop_price = signal.get("m15_triangle_stop")
        tp1_price = signal.get("m15_triangle_tp1")
        tp2_price = signal.get("m15_triangle_tp2")
        if pd.notna(stop_price):
            fig.add_hline(y=float(stop_price), line_color="#ef4444", line_dash="dot", annotation_text="初始止损", annotation_position="top right")
        if pd.notna(tp1_price):
            fig.add_hline(y=float(tp1_price), line_color="#0ea5e9", line_dash="dash", annotation_text="TP1", annotation_position="top right")
        if pd.notna(tp2_price):
            fig.add_hline(y=float(tp2_price), line_color="#22c55e", line_dash="dash", annotation_text="TP2", annotation_position="top right")

        touch_text = f'触碰: 顶={int(signal["m15_triangle_peak_touches"])} / 底={int(signal["m15_triangle_valley_touches"])}'
        note = signal.get("m15_triangle_note", "")
        stage = "正式信号" if review_mode == "signal" else "候选信号"
        fig.update_layout(
            title=(
                f"{symbol_label} | Trade {seq} | {direction} | {timestamp}<br>"
                f"{stage} | {touch_text} | note={note} | error={float(signal.get('m15_triangle_error', float('nan'))):.2f}"
            ),
            height=920,
            xaxis_rangeslider_visible=False,
        )
        fig.write_html(file_path)

        trade_index_rows.append(
            {
                "trade_id": str(seq),
                "timestamp": str(timestamp),
                "direction": direction,
                "file": filename,
                "peak_touches": str(int(signal["m15_triangle_peak_touches"])),
                "valley_touches": str(int(signal["m15_triangle_valley_touches"])),
                "mode": review_mode,
            }
        )
        links.append(f'<li><a href="{filename}">Trade {seq} | {direction} | {timestamp}</a></li>')

    index_html = (
        "<html><body>"
        f"<h1>{symbol_label} Triangle Review</h1>"
        f"<p>Mode: {review_mode}</p>"
        f"<p>Total signals: {len(trade_index_rows)}</p>"
        "<ul>"
        + "".join(links)
        + "</ul></body></html>"
    )
    index_path = review_dir / "index.html"
    index_path.write_text(index_html, encoding="utf-8")
    trade_index_path = review_dir / "trade_index.csv"
    pd.DataFrame(trade_index_rows).to_csv(trade_index_path, index=False)
    return {"review_dir": review_dir, "index": index_path, "trade_index": trade_index_path}


def export_structure_scan_review(
    candles_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    symbol_label: str,
    review_name: str = "structure_scan_review",
) -> dict[str, Path]:
    import plotly.graph_objects as go

    review_dir = settings.backtests_dir / review_name
    if review_dir.exists():
        shutil.rmtree(review_dir)
    review_dir.mkdir(parents=True, exist_ok=True)

    if candidates_df.empty:
        index_path = review_dir / "index.html"
        index_path.write_text("<html><body><h1>No structure candidates</h1></body></html>", encoding="utf-8")
        trade_index_path = review_dir / "trade_index.csv"
        pd.DataFrame(columns=["scan_id", "timestamp", "file"]).to_csv(trade_index_path, index=False)
        return {"review_dir": review_dir, "index": index_path, "trade_index": trade_index_path}

    rows: list[dict[str, str]] = []
    links: list[str] = []
    for seq, (_, row) in enumerate(candidates_df.iterrows(), start=1):
        timestamp = pd.to_datetime(row["timestamp"], utc=True)
        start = pd.to_datetime(row["window_start"], utc=True) - pd.Timedelta(hours=6)
        follow_end = pd.to_datetime(row.get("follow_end", pd.NaT), utc=True)
        if pd.isna(follow_end):
            follow_end = pd.to_datetime(row["window_end"], utc=True) + pd.Timedelta(hours=6)
        end = follow_end + pd.Timedelta(hours=2)
        view = candles_df[(candles_df["timestamp"] >= start) & (candles_df["timestamp"] <= end)].copy()
        if view.empty:
            continue

        filename = f"scan_{seq:03d}_{timestamp.strftime('%Y%m%d_%H%M')}.html"
        file_path = review_dir / filename

        fig = go.Figure()
        fig.add_trace(
            go.Candlestick(
                x=view["timestamp"],
                open=view["open"],
                high=view["high"],
                low=view["low"],
                close=view["close"],
                name="15m K",
            )
        )
        line_times = [pd.to_datetime(row["window_start"], utc=True), pd.to_datetime(row["window_end"], utc=True)]
        fig.add_trace(
            go.Scatter(
                x=line_times,
                y=[float(row["upper_start"]), float(row["upper_end"])],
                mode="lines",
                name="顶部连线",
                line={"color": "#2563eb", "width": 3},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=line_times,
                y=[float(row["lower_start"]), float(row["lower_end"])],
                mode="lines",
                name="底部连线",
                line={"color": "#1d4ed8", "width": 3},
                fill="tonexty",
                fillcolor="rgba(59,130,246,0.10)",
            )
        )
        projected_end = min(follow_end, pd.to_datetime(view["timestamp"].iloc[-1], utc=True))
        if projected_end > line_times[-1]:
            projection_times = [line_times[-1], projected_end]
            fig.add_trace(
                go.Scatter(
                    x=projection_times,
                    y=[float(row["upper_end"]), _project_structure_line(row, projected_end, "upper")],
                    mode="lines",
                    name="顶部延伸观察线",
                    line={"color": "#60a5fa", "width": 2, "dash": "dash"},
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=projection_times,
                    y=[float(row["lower_end"]), _project_structure_line(row, projected_end, "lower")],
                    mode="lines",
                    name="底部延伸观察线",
                    line={"color": "#3b82f6", "width": 2, "dash": "dash"},
                )
            )
        breakout_time = pd.to_datetime(row.get("breakout_time", pd.NaT), utc=True)
        breakout_direction = str(row.get("breakout_direction", "pending"))
        if pd.notna(breakout_time) and breakout_direction in {"up", "down"}:
            breakout_price = float(row["breakout_price"])
            fig.add_shape(
                type="line",
                x0=breakout_time,
                x1=breakout_time,
                y0=0,
                y1=1,
                xref="x",
                yref="paper",
                line={"color": "#f59e0b", "dash": "dash", "width": 2},
            )
            fig.add_annotation(
                x=breakout_time,
                y=1,
                xref="x",
                yref="paper",
                text="变盘确认",
                showarrow=False,
                yanchor="bottom",
            )
            fig.add_trace(
                go.Scatter(
                    x=[breakout_time],
                    y=[breakout_price],
                    mode="markers+text",
                    marker={
                        "size": 15,
                        "color": "#16a34a" if breakout_direction == "up" else "#dc2626",
                        "symbol": "triangle-up" if breakout_direction == "up" else "triangle-down",
                    },
                    text=["向上变盘" if breakout_direction == "up" else "向下变盘"],
                    textposition="top center" if breakout_direction == "up" else "bottom center",
                    name="通道变盘",
                )
            )
            if pd.notna(row.get("breakout_line_price", pd.NA)):
                fig.add_trace(
                    go.Scatter(
                        x=[breakout_time],
                        y=[float(row["breakout_line_price"])],
                        mode="markers",
                        marker={"size": 10, "color": "#f59e0b", "symbol": "x"},
                        name="突破线价格",
                    )
                )
        zone_half = float(row.get("zone_half_width", 0.0))
        if zone_half > 0:
            fig.add_trace(
                go.Scatter(
                    x=line_times,
                    y=[float(row["upper_start"]) + zone_half, float(row["upper_end"]) + zone_half],
                    mode="lines",
                    line={"color": "rgba(37,99,235,0.25)", "width": 1},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=line_times,
                    y=[float(row["upper_start"]) - zone_half, float(row["upper_end"]) - zone_half],
                    mode="lines",
                    line={"color": "rgba(37,99,235,0.25)", "width": 1},
                    fill="tonexty",
                    fillcolor="rgba(37,99,235,0.12)",
                    name="顶部触碰区域",
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=line_times,
                    y=[float(row["lower_start"]) + zone_half, float(row["lower_end"]) + zone_half],
                    mode="lines",
                    line={"color": "rgba(29,78,216,0.25)", "width": 1},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=line_times,
                    y=[float(row["lower_start"]) - zone_half, float(row["lower_end"]) - zone_half],
                    mode="lines",
                    line={"color": "rgba(29,78,216,0.25)", "width": 1},
                    fill="tonexty",
                    fillcolor="rgba(29,78,216,0.12)",
                    name="底部触碰区域",
                    hoverinfo="skip",
                )
            )
        fig.add_trace(
            go.Scatter(
                x=line_times,
                y=[float(row["mid_start"]), float(row["mid_end"])],
                mode="lines",
                name="中轨",
                line={"color": "#7c3aed", "width": 1.5, "dash": "dot"},
            )
        )

        peak_times = [pd.to_datetime(x) for x in row["peak_touch_times"]]
        valley_times = [pd.to_datetime(x) for x in row["valley_touch_times"]]
        fig.add_trace(
            go.Scatter(
                x=peak_times,
                y=list(row["peak_touch_prices"]),
                mode="markers",
                marker={"size": 10, "color": "#2563eb", "symbol": "circle"},
                name="顶部触碰",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=valley_times,
                y=list(row["valley_touch_prices"]),
                mode="markers",
                marker={"size": 10, "color": "#1d4ed8", "symbol": "circle"},
                name="底部触碰",
            )
        )
        breakout_label = "未走出通道"
        if breakout_direction == "up":
            breakout_label = f"向上变盘 {breakout_time}"
        elif breakout_direction == "down":
            breakout_label = f"向下变盘 {breakout_time}"

        fig.update_layout(
            title=(
                f"{symbol_label} | Scan {seq} | {timestamp}<br>"
                f"shape={row['shape']} | 顶触碰={int(row['peak_touches'])} | 底触碰={int(row['valley_touches'])} | "
                f"{breakout_label} | error={float(row['error']):.2f}"
            ),
            height=920,
            xaxis_rangeslider_visible=False,
        )
        fig.write_html(file_path)

        rows.append(
            {
                "scan_id": str(seq),
                "timestamp": str(timestamp),
                "shape": str(row["shape"]),
                "peak_touches": str(int(row["peak_touches"])),
                "valley_touches": str(int(row["valley_touches"])),
                "breakout_direction": breakout_direction,
                "breakout_time": "" if pd.isna(breakout_time) else str(breakout_time),
                "breakout_price": "" if pd.isna(row.get("breakout_price", pd.NA)) else f"{float(row['breakout_price']):.2f}",
                "file": filename,
            }
        )
        links.append(f'<li><a href="{filename}">Scan {seq} | {timestamp} | {row["shape"]} | {breakout_label}</a></li>')

    index_html = (
        "<html><body>"
        f"<h1>{symbol_label} Structure Scan Review</h1>"
        f"<p>Total candidates: {len(rows)}</p>"
        "<ul>"
        + "".join(links)
        + "</ul></body></html>"
    )
    index_path = review_dir / "index.html"
    index_path.write_text(index_html, encoding="utf-8")
    trade_index_path = review_dir / "trade_index.csv"
    pd.DataFrame(rows).to_csv(trade_index_path, index=False)
    return {"review_dir": review_dir, "index": index_path, "trade_index": trade_index_path}


def export_level_review(
    candles_df: pd.DataFrame,
    zones_df: pd.DataFrame,
    events_df: pd.DataFrame,
    symbol_label: str,
    review_name: str = "level_review",
    max_zones: int = 35,
) -> dict[str, Path]:
    import plotly.graph_objects as go

    review_dir = settings.backtests_dir / review_name
    if review_dir.exists():
        shutil.rmtree(review_dir)
    review_dir.mkdir(parents=True, exist_ok=True)

    index_path = review_dir / "index.html"
    zones_path = review_dir / "zones.csv"
    events_path = review_dir / "role_reversal_events.csv"

    zones_df.to_csv(zones_path, index=False)
    events_df.to_csv(events_path, index=False)

    if candles_df.empty or zones_df.empty:
        index_path.write_text("<html><body><h1>No support/resistance zones</h1></body></html>", encoding="utf-8")
        return {"review_dir": review_dir, "index": index_path, "zones": zones_path, "events": events_path}

    candles = candles_df.copy().sort_values("timestamp").reset_index(drop=True)
    candles["timestamp"] = pd.to_datetime(candles["timestamp"], utc=True)
    selected_zones = zones_df.sort_values("strength_score", ascending=False).head(max_zones).copy()
    selected_zones = selected_zones.sort_values("center").reset_index(drop=True)

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=candles["timestamp"],
            open=candles["open"],
            high=candles["high"],
            low=candles["low"],
            close=candles["close"],
            name="15m K",
        )
    )

    x0 = candles["timestamp"].iloc[0]
    x1 = candles["timestamp"].iloc[-1]
    for _, zone in selected_zones.iterrows():
        role = str(zone["current_role"])
        if role == "support":
            color = "rgba(34,197,94,0.16)"
            line_color = "rgba(34,197,94,0.60)"
        elif role == "resistance":
            color = "rgba(239,68,68,0.15)"
            line_color = "rgba(239,68,68,0.62)"
        else:
            color = "rgba(245,158,11,0.14)"
            line_color = "rgba(245,158,11,0.62)"

        fig.add_shape(
            type="rect",
            x0=x0,
            x1=x1,
            y0=float(zone["zone_low"]),
            y1=float(zone["zone_high"]),
            xref="x",
            yref="y",
            fillcolor=color,
            line={"color": line_color, "width": 1},
            layer="below",
        )
        fig.add_trace(
            go.Scatter(
                x=[x1],
                y=[float(zone["center"])],
                mode="markers+text",
                marker={"size": 7, "color": line_color},
                text=[
                    f"{zone['zone_id']} {role} 强度{float(zone['strength_score']):.1f} "
                    f"{zone['sources']}"
                ],
                textposition="middle right",
                name=f"{zone['zone_id']} {role}",
                hovertemplate=(
                    "zone=%{text}<br>"
                    f"low={float(zone['zone_low']):.2f}<br>"
                    f"high={float(zone['zone_high']):.2f}<br>"
                    f"touches={int(zone['touches'])}<extra></extra>"
                ),
            )
        )

    if not events_df.empty:
        events = events_df.copy()
        events = events[events["zone_id"].isin(set(selected_zones["zone_id"]))]
        events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True)
        up_events = events[events["direction"] == "up"]
        down_events = events[events["direction"] == "down"]
        if not up_events.empty:
            fig.add_trace(
                go.Scatter(
                    x=up_events["timestamp"],
                    y=up_events["breakout_price"],
                    mode="markers+text",
                    marker={"symbol": "triangle-up", "size": 12, "color": "#16a34a"},
                    text=up_events["zone_id"],
                    textposition="top center",
                    name="压力上破变支撑",
                    hovertext=up_events["reason"],
                )
            )
        if not down_events.empty:
            fig.add_trace(
                go.Scatter(
                    x=down_events["timestamp"],
                    y=down_events["breakout_price"],
                    mode="markers+text",
                    marker={"symbol": "triangle-down", "size": 12, "color": "#dc2626"},
                    text=down_events["zone_id"],
                    textposition="bottom center",
                    name="支撑跌破变压力",
                    hovertext=down_events["reason"],
                )
            )

        retests = events[events["status"] == "retest_confirmed"].dropna(subset=["retest_time", "retest_price"])
        if not retests.empty:
            fig.add_trace(
                go.Scatter(
                    x=pd.to_datetime(retests["retest_time"], utc=True),
                    y=retests["retest_price"],
                    mode="markers",
                    marker={"symbol": "x", "size": 10, "color": "#f59e0b"},
                    name="回踩确认",
                    hovertext=retests["reason"],
                )
            )

    fig.update_layout(
        title=(
            f"{symbol_label} Support / Resistance Map<br>"
            "绿色=当前支撑区，红色=当前压力区，黄色=当前价格附近活跃区；三角=支撑压力转换事件"
        ),
        height=960,
        xaxis_rangeslider_visible=False,
    )
    fig.write_html(index_path)
    return {"review_dir": review_dir, "index": index_path, "zones": zones_path, "events": events_path}


def export_level_snapshot_review(
    candles_df: pd.DataFrame,
    zones_df: pd.DataFrame,
    events_df: pd.DataFrame,
    symbol_label: str,
    as_of: pd.Timestamp,
    review_name: str = "level_snapshot_review",
) -> dict[str, Path]:
    import html

    import plotly.graph_objects as go

    review_dir = settings.backtests_dir / review_name
    if review_dir.exists():
        shutil.rmtree(review_dir)
    review_dir.mkdir(parents=True, exist_ok=True)

    index_path = review_dir / "index.html"
    zones_path = review_dir / "snapshot_zones.csv"
    events_path = review_dir / "snapshot_role_reversal_events.csv"
    zones_df.to_csv(zones_path, index=False)
    events_df.to_csv(events_path, index=False)

    if candles_df.empty:
        index_path.write_text("<html><body><h1>No candles for level snapshot</h1></body></html>", encoding="utf-8")
        return {"review_dir": review_dir, "index": index_path, "zones": zones_path, "events": events_path}

    candles = candles_df.copy().sort_values("timestamp").reset_index(drop=True)
    candles["timestamp"] = pd.to_datetime(candles["timestamp"], utc=True)
    snapshot_time = pd.to_datetime(as_of, utc=True)
    current_price = float(candles["close"].iloc[-1])

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=candles["timestamp"],
            open=candles["open"],
            high=candles["high"],
            low=candles["low"],
            close=candles["close"],
            name="15m K",
        )
    )

    x0 = candles["timestamp"].iloc[0]
    x1 = candles["timestamp"].iloc[-1]
    for _, zone in zones_df.iterrows():
        role = str(zone["current_role"])
        if role == "support":
            fill = "rgba(34,197,94,0.20)"
            line = "rgba(34,197,94,0.82)"
            label = "支撑"
        elif role == "resistance":
            fill = "rgba(239,68,68,0.18)"
            line = "rgba(239,68,68,0.84)"
            label = "压力"
        else:
            fill = "rgba(245,158,11,0.18)"
            line = "rgba(245,158,11,0.86)"
            label = "当前所在区域"

        fig.add_shape(
            type="rect",
            x0=x0,
            x1=x1,
            y0=float(zone["zone_low"]),
            y1=float(zone["zone_high"]),
            xref="x",
            yref="y",
            fillcolor=fill,
            line={"color": line, "width": 1.5},
            layer="below",
        )
        fig.add_trace(
            go.Scatter(
                x=[x1],
                y=[float(zone["center"])],
                mode="markers+text",
                marker={"size": 9, "color": line},
                text=[f"{zone['zone_id']} {label}"],
                textposition="middle right",
                name=f"{zone['zone_id']} {label}",
                hovertext=[str(zone.get("explanation", ""))],
            )
        )

    fig.add_hline(
        y=current_price,
        line_color="#f97316",
        line_dash="dash",
        annotation_text=f"当前价 {current_price:.2f}",
        annotation_position="top left",
    )
    fig.add_shape(
        type="line",
        x0=snapshot_time,
        x1=snapshot_time,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line={"color": "#f97316", "dash": "dash", "width": 2},
    )

    if not events_df.empty:
        events = events_df.copy()
        events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True)
        up_events = events[events["direction"] == "up"]
        down_events = events[events["direction"] == "down"]
        if not up_events.empty:
            fig.add_trace(
                go.Scatter(
                    x=up_events["timestamp"],
                    y=up_events["breakout_price"],
                    mode="markers+text",
                    marker={"symbol": "triangle-up", "size": 14, "color": "#16a34a"},
                    text=up_events["zone_id"],
                    textposition="top center",
                    name="刚完成: 压力变支撑",
                    hovertext=up_events["explanation"],
                )
            )
        if not down_events.empty:
            fig.add_trace(
                go.Scatter(
                    x=down_events["timestamp"],
                    y=down_events["breakout_price"],
                    mode="markers+text",
                    marker={"symbol": "triangle-down", "size": 14, "color": "#dc2626"},
                    text=down_events["zone_id"],
                    textposition="bottom center",
                    name="刚完成: 支撑变压力",
                    hovertext=down_events["explanation"],
                )
            )
        retests = events[events["status"] == "retest_confirmed"].dropna(subset=["retest_time", "retest_price"])
        if not retests.empty:
            fig.add_trace(
                go.Scatter(
                    x=pd.to_datetime(retests["retest_time"], utc=True),
                    y=retests["retest_price"],
                    mode="markers",
                    marker={"symbol": "x", "size": 11, "color": "#f59e0b"},
                    name="回踩确认",
                    hovertext=retests["explanation"],
                )
            )

    fig.update_layout(
        title=(
            f"{symbol_label} 支撑压力局部快照<br>"
            f"判断时间: {snapshot_time} | 当前价: {current_price:.2f} | 只展示附近关键区域"
        ),
        height=820,
        xaxis_rangeslider_visible=False,
    )

    figure_html = fig.to_html(full_html=False, include_plotlyjs=True)

    def zone_table() -> str:
        if zones_df.empty:
            return "<p>这一段没有识别到足够强的支撑压力区。</p>"
        rows: list[str] = []
        for _, zone in zones_df.iterrows():
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(zone['zone_id']))}</td>"
                f"<td>{html.escape(str(zone.get('role_cn', zone['current_role'])))}</td>"
                f"<td>{float(zone['zone_low']):.2f} - {float(zone['zone_high']):.2f}</td>"
                f"<td>{float(zone['distance_pct']) * 100:.2f}%</td>"
                f"<td>{html.escape(str(zone.get('source_cn', zone['sources'])))}</td>"
                f"<td>{html.escape(str(zone.get('explanation', '')))}</td>"
                "</tr>"
            )
        return (
            "<table>"
            "<thead><tr><th>区域</th><th>角色</th><th>价格区间</th><th>离当前价</th><th>来源</th><th>为什么</th></tr></thead>"
            "<tbody>"
            + "".join(rows)
            + "</tbody></table>"
        )

    def event_table() -> str:
        if events_df.empty:
            return "<p>最近没有识别到清晰的支撑压力转换。</p>"
        rows: list[str] = []
        for _, event in events_df.sort_values("timestamp", ascending=False).iterrows():
            retest = ""
            if pd.notna(event.get("retest_time", pd.NaT)):
                retest = f"{event['retest_time']} @ {float(event['retest_price']):.2f}"
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(event['timestamp']))}</td>"
                f"<td>{html.escape(str(event['zone_id']))}</td>"
                f"<td>{'压力变支撑' if event['direction'] == 'up' else '支撑变压力'}</td>"
                f"<td>{float(event['breakout_price']):.2f}</td>"
                f"<td>{html.escape(retest)}</td>"
                f"<td>{html.escape(str(event.get('explanation', '')))}</td>"
                "</tr>"
            )
        return (
            "<table>"
            "<thead><tr><th>时间</th><th>区域</th><th>转换</th><th>突破价</th><th>回踩确认</th><th>为什么</th></tr></thead>"
            "<tbody>"
            + "".join(rows)
            + "</tbody></table>"
        )

    html_doc = f"""
    <html>
    <head>
      <meta charset="utf-8" />
      <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #172033; }}
        h1, h2 {{ margin-bottom: 8px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; font-size: 14px; }}
        th, td {{ border: 1px solid #d7deea; padding: 8px 10px; vertical-align: top; }}
        th {{ background: #eef4ff; text-align: left; }}
        .note {{ color: #526078; }}
      </style>
    </head>
    <body>
      <h1>{html.escape(symbol_label)} 支撑压力局部判断</h1>
      <p class="note">这张图只回答当前这一段行情：哪里是支撑，哪里是压力，为什么，以及刚才哪里完成了支撑压力转换。</p>
      {figure_html}
      <h2>当前附近关键支撑压力</h2>
      {zone_table()}
      <h2>最近支撑压力转换</h2>
      {event_table()}
    </body>
    </html>
    """
    index_path.write_text(html_doc, encoding="utf-8")
    return {"review_dir": review_dir, "index": index_path, "zones": zones_path, "events": events_path}


def export_channel_level_strategy_review(
    candles_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    symbol_label: str,
    review_name: str = "channel_level_strategy_review",
) -> dict[str, Path]:
    import html

    import numpy as np
    import plotly.graph_objects as go

    review_dir = settings.backtests_dir / review_name
    if review_dir.exists():
        shutil.rmtree(review_dir)
    review_dir.mkdir(parents=True, exist_ok=True)

    index_path = review_dir / "index.html"
    trade_index_path = review_dir / "trade_index.csv"
    trades_df.to_csv(trade_index_path, index=False)

    if candles_df.empty or trades_df.empty:
        index_path.write_text("<html><body><h1>No channel-level strategy trades</h1></body></html>", encoding="utf-8")
        return {"review_dir": review_dir, "index": index_path, "trade_index": trade_index_path}

    candles = candles_df.copy().sort_values("timestamp").reset_index(drop=True)
    candles["timestamp"] = pd.to_datetime(candles["timestamp"], utc=True)
    rows: list[dict[str, str]] = []
    links: list[str] = []

    def _safe_float(value: object, default: float = float("nan")) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return default
        return result if pd.notna(result) else default

    def _list_value(value: object) -> list[object]:
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return []

    for seq, (_, trade) in enumerate(trades_df.iterrows(), start=1):
        direction = str(trade["direction"])
        entry_time = pd.to_datetime(trade["entry_time"], utc=True)
        exit_time = pd.to_datetime(trade["exit_time"], utc=True)
        window_start = pd.to_datetime(trade["window_start"], utc=True)
        window_end = pd.to_datetime(trade["window_end"], utc=True)
        view_start = min(window_start - pd.Timedelta(hours=4), entry_time - pd.Timedelta(hours=12))
        view_end = exit_time + pd.Timedelta(hours=8)
        view = candles[(candles["timestamp"] >= view_start) & (candles["timestamp"] <= view_end)].copy()
        if view.empty:
            continue

        filename = _safe_name(entry_time, direction, seq)
        file_path = review_dir / filename
        fig = go.Figure()
        fig.add_trace(
            go.Candlestick(
                x=view["timestamp"],
                open=view["open"],
                high=view["high"],
                low=view["low"],
                close=view["close"],
                name="15m K",
            )
        )

        line_times = [window_start, window_end]
        fig.add_trace(
            go.Scatter(
                x=line_times,
                y=[_safe_float(trade["upper_start"]), _safe_float(trade["upper_end"])],
                mode="lines",
                name="通道上沿",
                line={"color": "#2563eb", "width": 3},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=line_times,
                y=[_safe_float(trade["lower_start"]), _safe_float(trade["lower_end"])],
                mode="lines",
                name="通道下沿",
                line={"color": "#1d4ed8", "width": 3},
                fill="tonexty",
                fillcolor="rgba(37,99,235,0.10)",
            )
        )
        peak_times = [pd.to_datetime(x, utc=True) for x in _list_value(trade.get("peak_touch_times", []))]
        peak_prices = [_safe_float(x) for x in _list_value(trade.get("peak_touch_prices", []))]
        valley_times = [pd.to_datetime(x, utc=True) for x in _list_value(trade.get("valley_touch_times", []))]
        valley_prices = [_safe_float(x) for x in _list_value(trade.get("valley_touch_prices", []))]
        if peak_times and peak_prices:
            fig.add_trace(
                go.Scatter(
                    x=peak_times,
                    y=peak_prices,
                    mode="markers",
                    marker={"size": 9, "color": "#2563eb"},
                    name="顶部触碰",
                )
            )
        if valley_times and valley_prices:
            fig.add_trace(
                go.Scatter(
                    x=valley_times,
                    y=valley_prices,
                    mode="markers",
                    marker={"size": 9, "color": "#1d4ed8"},
                    name="底部触碰",
                )
            )

        x0 = view["timestamp"].iloc[0]
        x1 = view["timestamp"].iloc[-1]
        for zone_label, low_col, high_col, color in [
            ("TP1目标区", "tp1_zone_low", "tp1_zone_high", "rgba(245,158,11,0.16)"),
            ("TP2目标区", "tp2_zone_low", "tp2_zone_high", "rgba(34,197,94,0.14)"),
        ]:
            zone_low = _safe_float(trade.get(low_col))
            zone_high = _safe_float(trade.get(high_col))
            if np.isfinite(zone_low) and np.isfinite(zone_high):
                fig.add_shape(
                    type="rect",
                    x0=x0,
                    x1=x1,
                    y0=zone_low,
                    y1=zone_high,
                    xref="x",
                    yref="y",
                    fillcolor=color,
                    line={"color": color.replace("0.16", "0.65").replace("0.14", "0.65"), "width": 1},
                    layer="below",
                )
                fig.add_trace(
                    go.Scatter(
                        x=[x1],
                        y=[(zone_low + zone_high) / 2.0],
                        mode="text",
                        text=[zone_label],
                        textposition="middle right",
                        showlegend=False,
                    )
                )

        entry_price = _safe_float(trade["entry_price"])
        exit_price = _safe_float(trade["exit_price"])
        stop_price = _safe_float(trade["stop_loss_price"])
        tp1 = _safe_float(trade["tp1"])
        tp2 = _safe_float(trade["tp2"])
        marker_symbol = "triangle-up" if direction == "long" else "triangle-down"
        marker_color = "#16a34a" if direction == "long" else "#dc2626"

        fig.add_trace(
            go.Scatter(
                x=[entry_time],
                y=[entry_price],
                mode="markers+text",
                marker={"size": 15, "color": marker_color, "symbol": marker_symbol},
                text=["Entry"],
                textposition="top center" if direction == "long" else "bottom center",
                name="入场",
                hovertext=[str(trade.get("entry_reason", ""))],
            )
        )
        fig.add_hline(y=stop_price, line_color="#ef4444", line_dash="dot", annotation_text="初始止损", annotation_position="bottom right")
        fig.add_hline(y=tp1, line_color="#f59e0b", line_dash="dash", annotation_text="TP1 半仓", annotation_position="top right")
        fig.add_hline(y=tp2, line_color="#22c55e", line_dash="dash", annotation_text="TP2 全出", annotation_position="top right")
        fig.add_hline(y=entry_price, line_color="#a855f7", line_dash="dot", annotation_text="入场价参考线", annotation_position="bottom right")

        add_on_time = pd.to_datetime(trade.get("add_on_time", pd.NaT), utc=True)
        add_on_price = _safe_float(trade.get("add_on_entry_price"))
        if pd.notna(add_on_time) and np.isfinite(add_on_price):
            fig.add_trace(
                go.Scatter(
                    x=[add_on_time],
                    y=[add_on_price],
                    mode="markers+text",
                    marker={"size": 12, "color": "#0ea5e9", "symbol": "diamond"},
                    text=["补仓"],
                    textposition="top center" if direction == "long" else "bottom center",
                    name="回踩补仓",
                    hovertext=[str(trade.get("add_on_reason", ""))],
                )
            )

        tp1_time = pd.to_datetime(trade.get("tp1_time", pd.NaT), utc=True)
        if pd.notna(tp1_time):
            fig.add_trace(
                go.Scatter(
                    x=[tp1_time],
                    y=[tp1],
                    mode="markers+text",
                    marker={"size": 13, "color": "#f59e0b", "symbol": "x"},
                    text=["TP1 50%"],
                    textposition="top center",
                    name="TP1半仓",
                )
            )
        fig.add_trace(
            go.Scatter(
                x=[exit_time],
                y=[exit_price],
                mode="markers+text",
                marker={"size": 13, "color": "#111827", "symbol": "x"},
                text=[str(trade.get("exit_reason", ""))],
                textposition="bottom center",
                name="最终出场",
            )
        )

        pnl = _safe_float(trade.get("pnl_amount"))
        pnl_pct = _safe_float(trade.get("pnl_pct_equity")) * 100.0
        reverse_label = " | 反手单" if bool(trade.get("is_reverse", False)) else ""
        fig.update_layout(
            title=(
                f"{symbol_label} | Trade {seq} | {direction}{reverse_label} | {entry_time}<br>"
                f"入场理由: {trade.get('entry_reason', '')}<br>"
                f"出场: {trade.get('exit_reason', '')} | 盈亏: {pnl:.2f} USDT ({pnl_pct:.2f}%)"
            ),
            height=920,
            xaxis_rangeslider_visible=False,
        )
        figure_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
        add_on_status = "无合格回踩，不补仓"
        if pd.notna(add_on_time) and np.isfinite(add_on_price):
            add_on_status = f"{html.escape(str(add_on_time))} @ {add_on_price:.2f}，{html.escape(str(trade.get('add_on_reason', '')))}"

        table_html = (
            "<table>"
            "<tr><th>项目</th><th>内容</th></tr>"
            f"<tr><td>入场</td><td>{html.escape(str(entry_time))} @ {entry_price:.2f}</td></tr>"
            f"<tr><td>方向</td><td>{html.escape(direction)}</td></tr>"
            f"<tr><td>为什么进</td><td>{html.escape(str(trade.get('entry_reason', '')))}</td></tr>"
            f"<tr><td>补仓</td><td>{add_on_status}</td></tr>"
            f"<tr><td>止损逻辑</td><td>{html.escape(str(trade.get('stop_reason', '')))} SL={stop_price:.2f}</td></tr>"
            f"<tr><td>TP1</td><td>{tp1:.2f}，{html.escape(str(trade.get('tp1_zone_id', '')))}，半仓后止损不变，不开反手</td></tr>"
            f"<tr><td>TP2</td><td>{tp2:.2f}，{html.escape(str(trade.get('tp2_zone_id', '')))}，剩余仓位目标</td></tr>"
            f"<tr><td>目标依据</td><td>{html.escape(str(trade.get('target_reason', '')))}</td></tr>"
            f"<tr><td>通道触碰</td><td>顶 {int(_safe_float(trade.get('peak_touches'), 0))} 次，底 {int(_safe_float(trade.get('valley_touches'), 0))} 次</td></tr>"
            f"<tr><td>最终出场</td><td>{html.escape(str(exit_time))} @ {exit_price:.2f}，原因: {html.escape(str(trade.get('exit_reason', '')))}</td></tr>"
            f"<tr><td>盈亏</td><td>{pnl:.4f} USDT，权益变化 {pnl_pct:.2f}%</td></tr>"
            "</table>"
        )
        html_doc = f"""
        <html>
        <head>
          <meta charset="utf-8" />
          <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #172033; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 14px; font-size: 14px; }}
            th, td {{ border: 1px solid #d7deea; padding: 8px 10px; vertical-align: top; }}
            th {{ background: #eef4ff; text-align: left; }}
          </style>
        </head>
        <body>
          {figure_html}
          {table_html}
        </body>
        </html>
        """
        file_path.write_text(html_doc, encoding="utf-8")

        rows.append(
            {
                "trade_id": str(trade.get("trade_id", f"T{seq:04d}")),
                "direction": direction,
                "entry_time": str(entry_time),
                "exit_time": str(exit_time),
                "exit_reason": str(trade.get("exit_reason", "")),
                "pnl_amount": f"{pnl:.4f}",
                "pnl_pct_equity": f"{pnl_pct:.2f}%",
                "file": filename,
            }
        )
        links.append(
            f'<li><a href="{html.escape(filename)}">Trade {seq} | {html.escape(direction)} | '
            f'{html.escape(str(entry_time))} | {html.escape(str(trade.get("exit_reason", "")))} | {pnl:.2f} USDT</a></li>'
        )

    equity_tail = ""
    equity_chart = ""
    if not equity_df.empty:
        equity_plot = equity_df.copy()
        equity_plot["timestamp"] = pd.to_datetime(equity_plot["timestamp"], utc=True)
        final_equity = float(equity_plot["equity"].iloc[-1])
        peak = equity_plot["equity"].cummax()
        drawdown = equity_plot["equity"] / peak.replace(0, np.nan) - 1.0
        max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0
        equity_tail = f"<p>最终权益: {final_equity:.4f} USDT | 最大回撤: {max_drawdown * 100:.2f}%</p>"
        equity_fig = go.Figure()
        equity_fig.add_trace(
            go.Scatter(
                x=equity_plot["timestamp"],
                y=equity_plot["equity"],
                mode="lines",
                name="Equity",
                line={"color": "#2563eb", "width": 2},
            )
        )
        equity_fig.update_layout(
            title="权益曲线",
            height=360,
            xaxis_rangeslider_visible=False,
            yaxis_title="USDT",
        )
        equity_chart = equity_fig.to_html(full_html=False, include_plotlyjs="cdn")

    index_html = (
        "<html><body>"
        f"<h1>{html.escape(symbol_label)} 通道变盘 + 支撑压力策略逐单 Review</h1>"
        f"<p>交易数量: {len(rows)}</p>"
        f"{equity_tail}"
        f"{equity_chart}"
        "<ul>"
        + "".join(links)
        + "</ul></body></html>"
    )
    index_path.write_text(index_html, encoding="utf-8")
    pd.DataFrame(rows).to_csv(trade_index_path, index=False)
    return {"review_dir": review_dir, "index": index_path, "trade_index": trade_index_path}
