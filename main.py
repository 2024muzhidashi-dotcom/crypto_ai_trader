from __future__ import annotations

import argparse

import pandas as pd

from src.channel_level_strategy import ChannelLevelStrategyConfig, run_channel_level_strategy
from src.data_loader import OKXPerpDataLoader
from src.features import add_common_features
from src.levels import (
    LevelSnapshotConfig,
    SupportResistanceConfig,
    build_level_snapshot,
    detect_support_resistance_zones,
)
from src.market_structure import (
    TriangleDetectorConfig,
    StructureScanConfig,
    annotate_structure_breakouts,
    detect_triangle_reversals,
    merge_structure_candidates,
    scan_consolidation_structures,
)
from src.review import (
    clear_review_outputs,
    export_channel_level_strategy_review,
    export_level_review,
    export_level_snapshot_review,
    export_triangle_review,
    export_structure_scan_review,
)
from src.utils import save_json, symbol_to_filename


def fetch_symbol(symbol: str, months: int) -> None:
    loader = OKXPerpDataLoader()
    saved = loader.download_symbol_bundle(symbol, ["1d", "4h", "1h", "15m", "5m", "1m"], months)
    print(f"{symbol} saved: {saved}")


def reset_strategy_workspace() -> None:
    removed = clear_review_outputs()
    print("Old backtest/review outputs deleted:")
    for path in removed:
        print(path)


def detect_triangles(symbol: str, timeframe: str, lookback: int, recent_days: int | None = None) -> None:
    from src.data_loader import load_local_ohlcv
    from src.config import settings

    if timeframe != "15m":
        raise ValueError("Current clean-start detector first targets 15m only.")

    df = add_common_features(load_local_ohlcv(symbol, timeframe))
    if recent_days is not None and not df.empty:
        cutoff = pd.to_datetime(df["timestamp"].max(), utc=True) - pd.Timedelta(days=recent_days)
        df = df[df["timestamp"] >= cutoff].copy().reset_index(drop=True)
    prefixed = df.rename(columns={col: f"m15_{col}" for col in df.columns if col != "timestamp"})
    detected = detect_triangle_reversals(
        prefixed,
        "m15",
        TriangleDetectorConfig(lookback_bars=lookback),
    )
    output = detected[[
        "timestamp",
        "m15_open",
        "m15_high",
        "m15_low",
        "m15_close",
        "m15_ema20",
        "m15_ema50",
        "m15_triangle_upper",
        "m15_triangle_lower",
        "m15_triangle_mid",
        "m15_triangle_peak_touches",
        "m15_triangle_valley_touches",
        "m15_triangle_context",
        "m15_triangle_breakout_long",
        "m15_triangle_breakout_short",
        "m15_triangle_pullback_long",
        "m15_triangle_pullback_short",
        "m15_triangle_signal_long",
        "m15_triangle_signal_short",
        "m15_triangle_stop",
        "m15_triangle_tp1",
        "m15_triangle_tp2",
        "m15_triangle_error",
        "m15_triangle_note",
    ]].copy()
    path = settings.backtests_dir / f"{symbol_to_filename(symbol)}_{timeframe}_triangle_signals.csv"
    output.to_csv(path, index=False)
    print(f"Triangle detector output saved to {path}")
    try:
        review_paths = export_triangle_review(output, symbol_label=f"{symbol} {timeframe}", review_name=f"{symbol_to_filename(symbol)}_{timeframe}_triangle_review")
        print(f"Triangle review index saved to {review_paths['index']}")
        print(f"Triangle review csv saved to {review_paths['trade_index']}")
    except ModuleNotFoundError as exc:
        print(f"Triangle review skipped because dependency is missing: {exc}")
    print(output[(output["m15_triangle_signal_long"] == 1) | (output["m15_triangle_signal_short"] == 1)].tail(20).to_string(index=False))


def scan_structures(symbol: str, recent_days: int | None = None) -> None:
    from src.data_loader import load_local_ohlcv
    from src.config import settings

    df_15m = add_common_features(load_local_ohlcv(symbol, "15m"))
    if recent_days is not None:
        cutoff_15m = pd.to_datetime(df_15m["timestamp"].max(), utc=True) - pd.Timedelta(days=recent_days)
        df_15m = df_15m[df_15m["timestamp"] >= cutoff_15m].copy().reset_index(drop=True)

    candidates = scan_consolidation_structures(df_15m, StructureScanConfig())
    merged = merge_structure_candidates(candidates)
    tracked = annotate_structure_breakouts(df_15m, merged, StructureScanConfig())
    csv_path = settings.backtests_dir / f"{symbol_to_filename(symbol)}_15m_structure_candidates.csv"
    merged_csv_path = settings.backtests_dir / f"{symbol_to_filename(symbol)}_15m_structure_regions.csv"
    candidates.to_csv(csv_path, index=False)
    tracked.to_csv(merged_csv_path, index=False)
    review_paths = export_structure_scan_review(
        candles_df=df_15m,
        candidates_df=tracked,
        symbol_label=f"{symbol} 15m",
        review_name=f"{symbol_to_filename(symbol)}_15m_structure_review",
    )
    print(f"Structure candidates saved to {csv_path}")
    print(f"Structure regions saved to {merged_csv_path}")
    print(f"Structure review index saved to {review_paths['index']}")
    print(f"Structure review csv saved to {review_paths['trade_index']}")
    print(f"Raw candidates: {len(candidates)}")
    print(f"Merged structure regions: {len(tracked)}")


def scan_levels(symbol: str, timeframe: str = "15m", recent_days: int | None = None) -> None:
    from src.data_loader import load_local_ohlcv
    from src.config import settings

    df = add_common_features(load_local_ohlcv(symbol, timeframe))
    if recent_days is not None:
        cutoff = pd.to_datetime(df["timestamp"].max(), utc=True) - pd.Timedelta(days=recent_days)
        df = df[df["timestamp"] >= cutoff].copy().reset_index(drop=True)

    zones, events = detect_support_resistance_zones(df, SupportResistanceConfig())
    zones_path = settings.backtests_dir / f"{symbol_to_filename(symbol)}_{timeframe}_support_resistance_zones.csv"
    events_path = settings.backtests_dir / f"{symbol_to_filename(symbol)}_{timeframe}_role_reversal_events.csv"
    zones.to_csv(zones_path, index=False)
    events.to_csv(events_path, index=False)
    review_paths = export_level_review(
        candles_df=df,
        zones_df=zones,
        events_df=events,
        symbol_label=f"{symbol} {timeframe}",
        review_name=f"{symbol_to_filename(symbol)}_{timeframe}_level_review",
    )
    print(f"Support/resistance zones saved to {zones_path}")
    print(f"Role reversal events saved to {events_path}")
    print(f"Level review saved to {review_paths['index']}")
    print(f"Zones: {len(zones)}")
    print(f"Role reversal events: {len(events)}")
    if not zones.empty:
        print(zones.sort_values("strength_score", ascending=False).head(20).to_string(index=False))


def level_snapshot(
    symbol: str,
    timeframe: str = "15m",
    as_of: str | None = None,
    lookback_days: int = 14,
    recent_event_hours: int = 48,
) -> None:
    from src.data_loader import load_local_ohlcv

    df = add_common_features(load_local_ohlcv(symbol, timeframe))
    snapshot_time = pd.to_datetime(as_of, utc=True) if as_of else None
    context, selected_zones, recent_events, all_zones = build_level_snapshot(
        df,
        as_of=snapshot_time,
        sr_config=SupportResistanceConfig(),
        snapshot_config=LevelSnapshotConfig(
            lookback_days=lookback_days,
            recent_event_hours=recent_event_hours,
        ),
    )
    if context.empty:
        raise RuntimeError("No candles available for this snapshot.")

    actual_as_of = pd.to_datetime(context["timestamp"].iloc[-1], utc=True)
    review_paths = export_level_snapshot_review(
        candles_df=context,
        zones_df=selected_zones,
        events_df=recent_events,
        symbol_label=f"{symbol} {timeframe}",
        as_of=actual_as_of,
        review_name=f"{symbol_to_filename(symbol)}_{timeframe}_level_snapshot",
    )
    print(f"Level snapshot saved to {review_paths['index']}")
    print(f"Snapshot zones saved to {review_paths['zones']}")
    print(f"Snapshot role reversal events saved to {review_paths['events']}")
    print(f"Context candles: {len(context)}")
    print(f"All zones in context: {len(all_zones)}")
    print(f"Displayed zones: {len(selected_zones)}")
    print(f"Recent role reversal events: {len(recent_events)}")
    if not selected_zones.empty:
        print(
            selected_zones[
                [
                    "zone_id",
                    "role_cn",
                    "zone_low",
                    "zone_high",
                    "distance_pct",
                    "source_cn",
                    "strength_score",
                    "explanation",
                ]
            ].to_string(index=False)
        )
    if not recent_events.empty:
        print(
            recent_events[
                [
                    "timestamp",
                    "zone_id",
                    "direction",
                    "breakout_price",
                    "status",
                    "retest_time",
                    "retest_price",
                    "explanation",
                ]
            ].tail(10).to_string(index=False)
        )


def backtest_channel_levels(
    symbol: str,
    recent_days: int = 730,
    initial_capital: float = 10.0,
    position_pct: float = 0.05,
    leverage: float = 100.0,
) -> None:
    from src.config import settings
    from src.data_loader import load_local_ohlcv

    df_15m = add_common_features(load_local_ohlcv(symbol, "15m"))
    config = ChannelLevelStrategyConfig(
        initial_capital=initial_capital,
        position_pct=position_pct,
        leverage=leverage,
        recent_days=recent_days,
    )
    candles, signals, trades, equity, metrics = run_channel_level_strategy(df_15m, config)
    prefix = f"{symbol_to_filename(symbol)}_15m_channel_level_strategy"
    signals_path = settings.backtests_dir / f"{prefix}_signals.csv"
    trades_path = settings.backtests_dir / f"{prefix}_trades.csv"
    equity_path = settings.backtests_dir / f"{prefix}_equity.csv"
    metrics_path = settings.backtests_dir / f"{prefix}_metrics.json"
    signals.to_csv(signals_path, index=False)
    trades.to_csv(trades_path, index=False)
    equity.to_csv(equity_path, index=False)
    save_json(metrics, metrics_path)

    print(f"Signals generated: {len(signals)}", flush=True)
    print(f"Trades generated: {len(trades)}", flush=True)
    print("Exporting per-trade visual review...", flush=True)
    review_paths = export_channel_level_strategy_review(
        candles_df=candles,
        trades_df=trades,
        equity_df=equity,
        symbol_label=f"{symbol} 15m",
        review_name=f"{prefix}_review",
    )
    print(f"Channel-level strategy signals saved to {signals_path}")
    print(f"Channel-level strategy trades saved to {trades_path}")
    print(f"Channel-level strategy equity saved to {equity_path}")
    print(f"Channel-level strategy metrics saved to {metrics_path}")
    print(f"Channel-level strategy review saved to {review_paths['index']}")
    print(f"Signals: {len(signals)}")
    print(f"Trades: {len(trades)}")
    print(f"Initial capital: {metrics['initial_capital']:.2f} USDT")
    print(f"Final equity: {metrics['final_equity']:.4f} USDT")
    print(f"Net profit: {metrics['net_profit']:.4f} USDT")
    print(f"Return: {metrics['return_pct'] * 100:.2f}%")
    print(f"Win rate: {metrics['win_rate'] * 100:.2f}%")
    print(f"Profit factor: {metrics['profit_factor']}")
    print(f"Max drawdown: {metrics['max_drawdown'] * 100:.2f}%")
    print(f"Max consecutive losses: {metrics['max_consecutive_losses']}")
    if not trades.empty:
        print(
            trades[
                [
                    "trade_id",
                    "direction",
                    "entry_time",
                    "entry_price",
                    "exit_time",
                    "exit_price",
                    "exit_reason",
                    "pnl_amount",
                    "equity_after",
                ]
            ].tail(20).to_string(index=False)
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="crypto_ai_trader clean-slate runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser("fetch-symbol", help="Download OHLCV data for one symbol")
    fetch_parser.add_argument("--symbol", required=True)
    fetch_parser.add_argument("--months", type=int, default=36)

    subparsers.add_parser("reset-strategy-workspace", help="Delete old strategy review/backtest outputs")

    triangle_parser = subparsers.add_parser("detect-triangles", help="Run 15m contracting triangle detector")
    triangle_parser.add_argument("--symbol", required=True)
    triangle_parser.add_argument("--timeframe", default="15m")
    triangle_parser.add_argument("--lookback", type=int, default=100)
    triangle_parser.add_argument("--recent-days", type=int)

    scan_parser = subparsers.add_parser("scan-structures", help="Scan 15m consolidation structures only")
    scan_parser.add_argument("--symbol", required=True)
    scan_parser.add_argument("--recent-days", type=int)

    levels_parser = subparsers.add_parser("scan-levels", help="Scan support/resistance zones and role reversals")
    levels_parser.add_argument("--symbol", required=True)
    levels_parser.add_argument("--timeframe", default="15m")
    levels_parser.add_argument("--recent-days", type=int)

    snapshot_parser = subparsers.add_parser("level-snapshot", help="Explain nearby support/resistance for one market segment")
    snapshot_parser.add_argument("--symbol", required=True)
    snapshot_parser.add_argument("--timeframe", default="15m")
    snapshot_parser.add_argument("--as-of", help="UTC timestamp, e.g. '2026-05-06 04:45'. Defaults to latest candle.")
    snapshot_parser.add_argument("--lookback-days", type=int, default=14)
    snapshot_parser.add_argument("--recent-event-hours", type=int, default=48)

    channel_levels_parser = subparsers.add_parser(
        "backtest-channel-levels",
        help="Backtest channel breakout retest entries with support/resistance targets",
    )
    channel_levels_parser.add_argument("--symbol", required=True)
    channel_levels_parser.add_argument("--recent-days", type=int, default=730)
    channel_levels_parser.add_argument("--initial-capital", type=float, default=10.0)
    channel_levels_parser.add_argument("--position-pct", type=float, default=0.05)
    channel_levels_parser.add_argument("--leverage", type=float, default=100.0)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "fetch-symbol":
        fetch_symbol(args.symbol, args.months)
    elif args.command == "reset-strategy-workspace":
        reset_strategy_workspace()
    elif args.command == "detect-triangles":
        detect_triangles(args.symbol, args.timeframe, args.lookback, args.recent_days)
    elif args.command == "scan-structures":
        scan_structures(args.symbol, args.recent_days)
    elif args.command == "scan-levels":
        scan_levels(args.symbol, args.timeframe, args.recent_days)
    elif args.command == "level-snapshot":
        level_snapshot(args.symbol, args.timeframe, args.as_of, args.lookback_days, args.recent_event_hours)
    elif args.command == "backtest-channel-levels":
        backtest_channel_levels(args.symbol, args.recent_days, args.initial_capital, args.position_pct, args.leverage)


if __name__ == "__main__":
    main()
