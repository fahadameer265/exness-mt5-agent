"""Command line entry point.

Examples:
    python -m exness_mt5_agent check
    python -m exness_mt5_agent train
    python -m exness_mt5_agent backtest --symbol EURUSD
    python -m exness_mt5_agent live
"""

from __future__ import annotations

import argparse

from exness_mt5_agent.config import Settings
from exness_mt5_agent.pipeline import (
    configure_logging,
    run_backtest_symbol,
    run_check,
    run_live,
    train_symbols,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Exness MT5 research and demo execution agent")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="Connect, resolve symbols, and print account safety flags")

    train = commands.add_parser("train", help="Fit walk-forward models and save them")
    train.add_argument("--symbol", action="append", help="Override SYMBOLS. Repeat for several names.")
    train.add_argument("--bars", type=int, help="Closed M15 bars to request from MT5")
    train.add_argument("--min-setups", type=int, default=200, help="Minimum labeled setups per direction")

    backtest = commands.add_parser("backtest", help="Replay the saved model on historical bars")
    backtest.add_argument("--symbol", required=True)
    backtest.add_argument("--bars", type=int)
    backtest.add_argument("--equity", type=float, default=10_000.0)
    backtest.add_argument(
        "--rules-only",
        action="store_true",
        help="Trade every setup and skip the model. Use this to see the rule gate on its own.",
    )

    commands.add_parser("live", help="Poll closed M15 bars and log or send orders")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    settings = Settings.from_env()
    configure_logging(settings)
    if args.command == "check":
        run_check(settings)
        return
    if args.command == "train":
        symbols = args.symbol or list(settings.symbols)
        train_symbols(settings, symbols, args.bars or settings.history_bars, args.min_setups)
        return
    if args.command == "backtest":
        run_backtest_symbol(
            settings,
            args.symbol,
            args.bars or settings.history_bars,
            args.equity,
            args.rules_only,
        )
        return
    if args.command == "live":
        run_live(settings)
        return
    raise SystemExit(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
