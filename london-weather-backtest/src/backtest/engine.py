"""Main backtest engine that combines data, buckets, and strategies."""

from __future__ import annotations

import logging

import pandas as pd

from config import DATA_PROCESSED_DIR, DEFAULT_PRICE_SCENARIO
from src.backtest.buckets import generate_daily_buckets
from src.backtest.strategy import (
    compute_trailing_bucket_accuracy,
    strategy_best_model,
    strategy_convergence,
    strategy_early_convergence,
    strategy_forced_bet,
    strategy_naive_model_follow,
)

logger = logging.getLogger(__name__)


def build_daily_dataframe(
    forecasts: pd.DataFrame,
    actuals: pd.DataFrame,
    forecasts_j2: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge forecasts and actuals into a single daily DataFrame.

    Computes error columns for each model and lead time.

    Args:
        forecasts: DataFrame with date + model forecast columns (J-1).
        actuals: DataFrame with date + actual_temp column.
        forecasts_j2: Optional DataFrame with J-2 forecasts.

    Returns:
        Merged DataFrame with error columns added.
    """
    df = forecasts.merge(actuals, on="date", how="inner")

    for model in ["gfs", "ecmwf"]:
        col = f"{model}_forecast"
        if col in df.columns:
            df[f"{model}_error"] = df[col] - df["actual_temp"]

    # Merge J-2 forecasts if available
    if forecasts_j2 is not None:
        # Rename J-2 columns to avoid clash with J-1
        j2_renamed = forecasts_j2.rename(columns={
            c: c.replace("_forecast", "_forecast_j2")
            for c in forecasts_j2.columns if c != "date"
        })
        df = df.merge(j2_renamed, on="date", how="left")

        for model in ["gfs", "ecmwf"]:
            col = f"{model}_forecast_j2"
            if col in df.columns:
                df[f"{model}_error_j2"] = df[col] - df["actual_temp"]

    df = df.sort_values("date").reset_index(drop=True)
    return df


def run_backtest(
    daily_df: pd.DataFrame,
    strategies: list[str] | None = None,
    scenario: str = DEFAULT_PRICE_SCENARIO,
) -> pd.DataFrame:
    """Run the full backtest over the daily data.

    Args:
        daily_df: DataFrame from build_daily_dataframe().
        strategies: List of strategy names to run. None = all.
        scenario: Price scenario key.

    Returns:
        DataFrame of all trades (trades log).
    """
    all_strategies = [
        "naive_gfs", "naive_ecmwf", "best_model", "convergence",
        "early_conv_36h", "early_conv_18h", "forced_bet",
    ]
    if strategies is None:
        strategies = all_strategies

    # Check if J-2 columns are available for early entry strategies
    has_j2 = "gfs_forecast_j2" in daily_df.columns and "ecmwf_forecast_j2" in daily_df.columns

    all_trades: list[dict] = []
    naive_gfs_history: list[dict] = []
    naive_ecmwf_history: list[dict] = []

    logger.info("Running backtest with strategies: %s", strategies)

    for _, row in daily_df.iterrows():
        month = row["date"].month
        buckets = generate_daily_buckets(month)

        # Strategy 1: Naive GFS
        if "naive_gfs" in strategies:
            trade = strategy_naive_model_follow(row, buckets, model="gfs", scenario=scenario)
            if trade is not None:
                all_trades.append(trade)
                naive_gfs_history.append(trade)

        # Strategy 1b: Naive ECMWF
        if "naive_ecmwf" in strategies:
            trade = strategy_naive_model_follow(row, buckets, model="ecmwf", scenario=scenario)
            if trade is not None:
                all_trades.append(trade)
                naive_ecmwf_history.append(trade)

        # Strategy 2: Best Model
        if "best_model" in strategies:
            trailing_acc = compute_trailing_bucket_accuracy(
                naive_gfs_history + naive_ecmwf_history
            )
            trade = strategy_best_model(row, buckets, trailing_acc, scenario=scenario)
            if trade is not None:
                all_trades.append(trade)

        # Strategy 3: Convergence (J-1, ~18h before, fixed pricing)
        if "convergence" in strategies:
            trade = strategy_convergence(row, buckets, scenario=scenario)
            if trade is not None:
                all_trades.append(trade)

        # Strategy 4: Early convergence J-1 evening (~18h, time-decay pricing)
        if "early_conv_18h" in strategies:
            trade = strategy_early_convergence(row, buckets, lead_hours=18)
            if trade is not None:
                all_trades.append(trade)

        # Strategy 5: Early convergence J-2 evening (~36h, time-decay pricing)
        if "early_conv_36h" in strategies and has_j2:
            trade = strategy_early_convergence(
                row, buckets, lead_hours=36,
                gfs_col="gfs_forecast_j2", ecmwf_col="ecmwf_forecast_j2",
            )
            if trade is not None:
                all_trades.append(trade)

        # Strategy 6: Forced bet — trade EVERY day (no survivorship bias)
        if "forced_bet" in strategies:
            trailing_acc = compute_trailing_bucket_accuracy(
                naive_gfs_history + naive_ecmwf_history
            )
            trade = strategy_forced_bet(row, buckets, trailing_acc, scenario=scenario)
            if trade is not None:
                all_trades.append(trade)

    trades_df = pd.DataFrame(all_trades)
    logger.info("Backtest complete: %d total trades", len(trades_df))

    return trades_df


def save_processed_data(
    daily_df: pd.DataFrame,
    trades_df: pd.DataFrame,
) -> None:
    """Save processed DataFrames to CSV.

    Args:
        daily_df: The daily forecasts + actuals DataFrame.
        trades_df: The trades log DataFrame.
    """
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    daily_path = DATA_PROCESSED_DIR / "forecasts_daily.csv"
    daily_df.to_csv(daily_path, index=False)
    logger.info("Saved daily data to %s", daily_path)

    trades_path = DATA_PROCESSED_DIR / "trades_log.csv"
    trades_df.to_csv(trades_path, index=False)
    logger.info("Saved trades log to %s", trades_path)
