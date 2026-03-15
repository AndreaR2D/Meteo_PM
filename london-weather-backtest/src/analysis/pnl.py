"""P/L and trading performance metrics."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_strategy_summary(trades_df: pd.DataFrame) -> dict[str, dict]:
    """Compute summary metrics for each strategy.

    Args:
        trades_df: DataFrame from the backtest engine.

    Returns:
        Dict mapping strategy name to metrics dict.
    """
    summaries = {}

    # Group by strategy base name (normalize best_model variants)
    trades = trades_df.copy()
    trades["strategy_base"] = trades["strategy"].apply(
        lambda s: "best_model" if s.startswith("best_model") else s
    )

    for strategy, group in trades.groupby("strategy_base"):
        total = len(group)
        wins = group["win"].sum()
        win_rate = wins / total if total > 0 else 0.0
        total_pnl = group["pnl"].sum()

        # Monthly P/L
        group = group.copy()
        group["month"] = pd.to_datetime(group["date"]).dt.to_period("M")
        monthly_pnl = group.groupby("month")["pnl"].sum()

        # Max drawdown
        cumulative = group["pnl"].cumsum()
        running_max = cumulative.cummax()
        drawdown = cumulative - running_max
        max_drawdown = drawdown.min()

        # Sharpe ratio (daily, annualized)
        daily_returns = group.groupby(pd.to_datetime(group["date"]).dt.date)["pnl"].sum()
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Best/worst month
        best_month = monthly_pnl.idxmax() if len(monthly_pnl) > 0 else None
        worst_month = monthly_pnl.idxmin() if len(monthly_pnl) > 0 else None

        summaries[strategy] = {
            "total_trades": total,
            "wins": int(wins),
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe,
            "best_month": str(best_month) if best_month is not None else "N/A",
            "best_month_pnl": monthly_pnl.max() if len(monthly_pnl) > 0 else 0.0,
            "worst_month": str(worst_month) if worst_month is not None else "N/A",
            "worst_month_pnl": monthly_pnl.min() if len(monthly_pnl) > 0 else 0.0,
            "monthly_pnl": monthly_pnl.to_dict(),
        }

    return summaries


def compute_convergence_analysis(trades_df: pd.DataFrame) -> dict:
    """Analyze win rate when models converge vs when they diverge.

    Uses the convergence strategy trades and naive trades for comparison.

    Args:
        trades_df: DataFrame from the backtest engine.

    Returns:
        Dict with convergence analysis metrics.
    """
    convergence_trades = trades_df[trades_df["strategy"] == "convergence"]
    naive_gfs_trades = trades_df[trades_df["strategy"] == "naive_gfs"]

    # Days where convergence strategy traded = models agreed
    convergence_dates = set(convergence_trades["date"].dt.date)
    # Days where GFS traded but convergence didn't = models disagreed
    gfs_dates = set(naive_gfs_trades["date"].dt.date)
    divergence_dates = gfs_dates - convergence_dates

    # Win rate on convergence days (from convergence strategy)
    conv_total = len(convergence_trades)
    conv_wins = convergence_trades["win"].sum() if conv_total > 0 else 0
    conv_win_rate = conv_wins / conv_total if conv_total > 0 else 0.0

    # Win rate on divergence days (from naive GFS, as baseline)
    divergence_trades = naive_gfs_trades[
        naive_gfs_trades["date"].dt.date.isin(divergence_dates)
    ]
    div_total = len(divergence_trades)
    div_wins = divergence_trades["win"].sum() if div_total > 0 else 0
    div_win_rate = div_wins / div_total if div_total > 0 else 0.0

    return {
        "convergence_days": conv_total,
        "convergence_wins": int(conv_wins),
        "convergence_win_rate": conv_win_rate,
        "divergence_days": div_total,
        "divergence_wins": int(div_wins),
        "divergence_win_rate": div_win_rate,
    }


def compute_monthly_summary(
    daily_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    model_metrics: dict,
) -> pd.DataFrame:
    """Build a monthly summary DataFrame.

    Args:
        daily_df: Daily forecasts + actuals.
        trades_df: Trades log.
        model_metrics: Dict from compute_all_metrics.

    Returns:
        DataFrame with monthly rows and summary columns.
    """
    months = sorted(daily_df["date"].dt.month.unique())
    rows = []

    trades = trades_df.copy()
    trades["strategy_base"] = trades["strategy"].apply(
        lambda s: "best_model" if s.startswith("best_model") else s
    )
    trades["month"] = pd.to_datetime(trades["date"]).dt.month

    for month in months:
        row = {"month": month}

        for model in ["gfs", "ecmwf"]:
            if model in model_metrics:
                row[f"{model}_mae"] = model_metrics[model]["monthly_mae"].get(month, None)
                row[f"{model}_bucket_acc"] = model_metrics[model]["monthly_bucket_accuracy"].get(month, None)

        for strategy in ["naive_gfs", "naive_ecmwf", "best_model", "convergence"]:
            month_trades = trades[
                (trades["month"] == month) & (trades["strategy_base"] == strategy)
            ]
            row[f"pnl_{strategy}"] = month_trades["pnl"].sum() if len(month_trades) > 0 else 0.0

        rows.append(row)

    return pd.DataFrame(rows)
