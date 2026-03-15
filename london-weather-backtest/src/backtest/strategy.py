"""Trading strategies for the weather backtest."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config import (
    DEFAULT_PRICE_SCENARIO,
    PRICE_SCENARIOS,
    TIME_DECAY_PRICING,
    TRAILING_WINDOW_DAYS,
)
from src.backtest.buckets import assign_bucket

logger = logging.getLogger(__name__)


def _get_buy_price(
    bet_bucket_idx: int,
    consensus_bucket_idx: int,
    scenario: str = DEFAULT_PRICE_SCENARIO,
) -> float:
    """Determine the simulated buy price based on bucket distance from consensus.

    Args:
        bet_bucket_idx: Index of the bucket we're betting on.
        consensus_bucket_idx: Index of the consensus bucket.
        scenario: Price scenario key from PRICE_SCENARIOS.

    Returns:
        Simulated buy price in dollars.
    """
    prices = PRICE_SCENARIOS[scenario]
    distance = abs(bet_bucket_idx - consensus_bucket_idx)

    if distance == 0:
        return prices["consensus_bucket"]
    elif distance == 1:
        return prices["adjacent_bucket"]
    else:
        return prices["far_bucket"]


def strategy_naive_model_follow(
    row: pd.Series,
    buckets: list[dict],
    model: str = "gfs",
    scenario: str = DEFAULT_PRICE_SCENARIO,
) -> dict | None:
    """Strategy 1: Naive Model Follow — bet on the bucket predicted by a single model.

    Args:
        row: DataFrame row with forecast and actual data.
        buckets: List of bucket dicts for this day.
        model: Which model to follow ('gfs' or 'ecmwf').
        scenario: Price scenario for simulated prices.

    Returns:
        Trade dict or None if no valid trade.
    """
    forecast_col = f"{model}_forecast"
    forecast = row.get(forecast_col)
    actual = row.get("actual_temp")

    if pd.isna(forecast) or pd.isna(actual):
        return None

    predicted_bucket = assign_bucket(forecast, buckets)
    actual_bucket = assign_bucket(actual, buckets)

    if predicted_bucket is None or actual_bucket is None:
        return None

    buy_price = _get_buy_price(
        predicted_bucket["index"], predicted_bucket["index"], scenario
    )
    win = predicted_bucket["index"] == actual_bucket["index"]
    pnl = (1.0 - buy_price) if win else -buy_price

    return {
        "date": row["date"],
        "strategy": f"naive_{model}",
        "bucket_bet": predicted_bucket["label"],
        "bucket_bet_idx": predicted_bucket["index"],
        "price": buy_price,
        "actual_bucket": actual_bucket["label"],
        "actual_bucket_idx": actual_bucket["index"],
        "win": win,
        "pnl": pnl,
    }


def strategy_best_model(
    row: pd.Series,
    buckets: list[dict],
    trailing_accuracy: dict[str, float],
    scenario: str = DEFAULT_PRICE_SCENARIO,
) -> dict | None:
    """Strategy 2: Best Model Follow — bet using the model with best trailing accuracy.

    Args:
        row: DataFrame row with forecast and actual data.
        buckets: List of bucket dicts for this day.
        trailing_accuracy: Dict mapping model name to trailing bucket accuracy.
        scenario: Price scenario for simulated prices.

    Returns:
        Trade dict or None if no valid trade.
    """
    # Pick the model with the best trailing bucket accuracy
    best_model = max(trailing_accuracy, key=trailing_accuracy.get)

    result = strategy_naive_model_follow(row, buckets, model=best_model, scenario=scenario)
    if result is not None:
        result["strategy"] = f"best_model({best_model})"
    return result


def strategy_convergence(
    row: pd.Series,
    buckets: list[dict],
    scenario: str = DEFAULT_PRICE_SCENARIO,
) -> dict | None:
    """Strategy 3: Convergence Bet — only bet when GFS and ECMWF agree on the same bucket.

    Args:
        row: DataFrame row with forecast and actual data.
        buckets: List of bucket dicts for this day.
        scenario: Price scenario for simulated prices.

    Returns:
        Trade dict or None if models diverge or data is missing.
    """
    gfs_forecast = row.get("gfs_forecast")
    ecmwf_forecast = row.get("ecmwf_forecast")
    actual = row.get("actual_temp")

    if pd.isna(gfs_forecast) or pd.isna(ecmwf_forecast) or pd.isna(actual):
        return None

    gfs_bucket = assign_bucket(gfs_forecast, buckets)
    ecmwf_bucket = assign_bucket(ecmwf_forecast, buckets)
    actual_bucket = assign_bucket(actual, buckets)

    if gfs_bucket is None or ecmwf_bucket is None or actual_bucket is None:
        return None

    # Only trade when models converge (same bucket)
    if gfs_bucket["index"] != ecmwf_bucket["index"]:
        return None

    consensus_idx = gfs_bucket["index"]
    buy_price = _get_buy_price(consensus_idx, consensus_idx, scenario)
    win = consensus_idx == actual_bucket["index"]
    pnl = (1.0 - buy_price) if win else -buy_price

    return {
        "date": row["date"],
        "strategy": "convergence",
        "bucket_bet": gfs_bucket["label"],
        "bucket_bet_idx": consensus_idx,
        "price": buy_price,
        "actual_bucket": actual_bucket["label"],
        "actual_bucket_idx": actual_bucket["index"],
        "win": win,
        "pnl": pnl,
    }


def _get_timedecay_price(
    bet_bucket_idx: int,
    consensus_bucket_idx: int,
    lead_hours: int,
) -> float:
    """Get simulated buy price using the time-decay model.

    Interpolates between the configured price tiers based on lead_hours.
    Earlier entry → lower price → more alpha.

    Args:
        bet_bucket_idx: Index of the bucket we're betting on.
        consensus_bucket_idx: Index of the consensus bucket.
        lead_hours: Hours before resolution when the bet is placed.

    Returns:
        Simulated buy price in dollars.
    """
    distance = abs(bet_bucket_idx - consensus_bucket_idx)
    if distance == 0:
        key = "consensus_bucket"
    elif distance == 1:
        key = "adjacent_bucket"
    else:
        key = "far_bucket"

    # Sort tiers by lead hours
    tiers = sorted(TIME_DECAY_PRICING.items())  # [(2, {...}), (6, {...}), ...]

    # Clamp to bounds
    if lead_hours <= tiers[0][0]:
        return tiers[0][1][key]
    if lead_hours >= tiers[-1][0]:
        return tiers[-1][1][key]

    # Linear interpolation between surrounding tiers
    for i in range(len(tiers) - 1):
        h_low, p_low = tiers[i][0], tiers[i][1][key]
        h_high, p_high = tiers[i + 1][0], tiers[i + 1][1][key]
        if h_low <= lead_hours <= h_high:
            ratio = (lead_hours - h_low) / (h_high - h_low)
            return p_low + ratio * (p_high - p_low)

    return tiers[-1][1][key]


def strategy_early_convergence(
    row: pd.Series,
    buckets: list[dict],
    lead_hours: int = 36,
    gfs_col: str = "gfs_forecast",
    ecmwf_col: str = "ecmwf_forecast",
) -> dict | None:
    """Convergence bet with time-decay pricing.

    Same logic as convergence, but the buy price depends on how early
    the bet is placed (lead_hours). J-2 evening = 36h, J-1 evening = 18h.

    Args:
        row: DataFrame row with forecast and actual data.
        buckets: List of bucket dicts for this day.
        lead_hours: Hours before resolution when bet is placed.
        gfs_col: Column name for GFS forecast.
        ecmwf_col: Column name for ECMWF forecast.

    Returns:
        Trade dict or None if models diverge or data is missing.
    """
    gfs_forecast = row.get(gfs_col)
    ecmwf_forecast = row.get(ecmwf_col)
    actual = row.get("actual_temp")

    if pd.isna(gfs_forecast) or pd.isna(ecmwf_forecast) or pd.isna(actual):
        return None

    gfs_bucket = assign_bucket(gfs_forecast, buckets)
    ecmwf_bucket = assign_bucket(ecmwf_forecast, buckets)
    actual_bucket = assign_bucket(actual, buckets)

    if gfs_bucket is None or ecmwf_bucket is None or actual_bucket is None:
        return None

    # Only trade when models converge
    if gfs_bucket["index"] != ecmwf_bucket["index"]:
        return None

    consensus_idx = gfs_bucket["index"]
    buy_price = _get_timedecay_price(consensus_idx, consensus_idx, lead_hours)
    win = consensus_idx == actual_bucket["index"]
    pnl = (1.0 - buy_price) if win else -buy_price

    return {
        "date": row["date"],
        "strategy": f"early_conv_{lead_hours}h",
        "bucket_bet": gfs_bucket["label"],
        "bucket_bet_idx": consensus_idx,
        "price": buy_price,
        "actual_bucket": actual_bucket["label"],
        "actual_bucket_idx": actual_bucket["index"],
        "win": win,
        "pnl": pnl,
        "lead_hours": lead_hours,
    }


def strategy_forced_bet(
    row: pd.Series,
    buckets: list[dict],
    trailing_accuracy: dict[str, float],
    scenario: str = DEFAULT_PRICE_SCENARIO,
) -> dict | None:
    """Forced bet strategy — must trade every day, pick best model.

    When models converge: bet on consensus bucket (same as convergence).
    When models diverge: bet on the model with best trailing accuracy.
    This removes survivorship bias by trading ALL days.

    Args:
        row: DataFrame row with forecast and actual data.
        buckets: List of bucket dicts for this day.
        trailing_accuracy: Dict mapping model name to trailing bucket accuracy.
        scenario: Price scenario for simulated prices.

    Returns:
        Trade dict or None if data is missing.
    """
    gfs_forecast = row.get("gfs_forecast")
    ecmwf_forecast = row.get("ecmwf_forecast")
    actual = row.get("actual_temp")

    if pd.isna(gfs_forecast) or pd.isna(ecmwf_forecast) or pd.isna(actual):
        return None

    gfs_bucket = assign_bucket(gfs_forecast, buckets)
    ecmwf_bucket = assign_bucket(ecmwf_forecast, buckets)
    actual_bucket = assign_bucket(actual, buckets)

    if gfs_bucket is None or ecmwf_bucket is None or actual_bucket is None:
        return None

    converge = gfs_bucket["index"] == ecmwf_bucket["index"]

    if converge:
        # Models agree — bet on consensus
        bet_bucket = gfs_bucket
        sub_strategy = "conv"
    else:
        # Models disagree — pick the one with best trailing accuracy
        best = max(trailing_accuracy, key=trailing_accuracy.get)
        bet_bucket = gfs_bucket if best == "gfs" else ecmwf_bucket
        sub_strategy = f"div_{best}"

    buy_price = _get_buy_price(bet_bucket["index"], bet_bucket["index"], scenario)
    win = bet_bucket["index"] == actual_bucket["index"]
    pnl = (1.0 - buy_price) if win else -buy_price

    return {
        "date": row["date"],
        "strategy": f"forced_bet({sub_strategy})",
        "bucket_bet": bet_bucket["label"],
        "bucket_bet_idx": bet_bucket["index"],
        "price": buy_price,
        "actual_bucket": actual_bucket["label"],
        "actual_bucket_idx": actual_bucket["index"],
        "win": win,
        "pnl": pnl,
        "converge": converge,
    }


def strategy_price_threshold(
    row: pd.Series,
    buckets: list[dict],
    market_price: float,
    max_price: float = 0.60,
) -> dict | None:
    """Convergence bet with a price ceiling — only enter if price is below threshold.

    This is the realistic strategy: check convergence, check market price,
    only bet if the price offers enough edge.

    Args:
        row: DataFrame row with forecast and actual data.
        buckets: List of bucket dicts for this day.
        market_price: The current market price for the consensus bucket.
        max_price: Maximum price we're willing to pay. Skip if price > max_price.

    Returns:
        Trade dict or None if models diverge, data missing, or price too high.
    """
    gfs_forecast = row.get("gfs_forecast")
    ecmwf_forecast = row.get("ecmwf_forecast")
    actual = row.get("actual_temp")

    if pd.isna(gfs_forecast) or pd.isna(ecmwf_forecast) or pd.isna(actual):
        return None

    gfs_bucket = assign_bucket(gfs_forecast, buckets)
    ecmwf_bucket = assign_bucket(ecmwf_forecast, buckets)
    actual_bucket = assign_bucket(actual, buckets)

    if gfs_bucket is None or ecmwf_bucket is None or actual_bucket is None:
        return None

    # Only trade when models converge
    if gfs_bucket["index"] != ecmwf_bucket["index"]:
        return None

    # Only trade when price is below threshold
    if market_price > max_price:
        return None

    consensus_idx = gfs_bucket["index"]
    win = consensus_idx == actual_bucket["index"]
    # Apply 2% fee on winnings (Polymarket fee structure)
    gross_gain = (1.0 - market_price)
    fee = gross_gain * 0.02 if win else 0.0
    pnl = (gross_gain - fee) if win else -market_price

    return {
        "date": row["date"],
        "strategy": f"threshold_{max_price:.0%}",
        "bucket_bet": gfs_bucket["label"],
        "bucket_bet_idx": consensus_idx,
        "price": market_price,
        "actual_bucket": actual_bucket["label"],
        "actual_bucket_idx": actual_bucket["index"],
        "win": win,
        "pnl": pnl,
    }


def compute_trailing_bucket_accuracy(
    trades_history: list[dict],
    window: int = TRAILING_WINDOW_DAYS,
) -> dict[str, float]:
    """Compute trailing bucket accuracy for each model over the last N days.

    Args:
        trades_history: List of past trade dicts (from naive strategies).
        window: Number of past days to consider.

    Returns:
        Dict mapping model key to accuracy (0-1). Defaults to 0.5 if no history.
    """
    accuracy = {}

    for model in ["gfs", "ecmwf"]:
        model_trades = [
            t for t in trades_history
            if t["strategy"] == f"naive_{model}"
        ]
        recent = model_trades[-window:] if len(model_trades) > window else model_trades

        if not recent:
            accuracy[model] = 0.5
        else:
            wins = sum(1 for t in recent if t["win"])
            accuracy[model] = wins / len(recent)

    return accuracy
