"""Model accuracy analysis: MAE, bucket accuracy, bias."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.backtest.buckets import assign_bucket, generate_daily_buckets

logger = logging.getLogger(__name__)


def compute_mae(daily_df: pd.DataFrame, model: str) -> float:
    """Compute Mean Absolute Error for a model.

    Args:
        daily_df: Daily DataFrame with error columns.
        model: Model key ('gfs' or 'ecmwf').

    Returns:
        MAE in °C.
    """
    error_col = f"{model}_error"
    return daily_df[error_col].abs().mean()


def compute_monthly_mae(daily_df: pd.DataFrame, model: str) -> pd.Series:
    """Compute MAE per month for a model.

    Args:
        daily_df: Daily DataFrame with error columns.
        model: Model key ('gfs' or 'ecmwf').

    Returns:
        Series indexed by month with MAE values.
    """
    error_col = f"{model}_error"
    df = daily_df.copy()
    df["month"] = df["date"].dt.month
    return df.groupby("month")[error_col].apply(lambda x: x.abs().mean())


def compute_bias(daily_df: pd.DataFrame, model: str) -> float:
    """Compute mean bias (positive = overestimates, negative = underestimates).

    Args:
        daily_df: Daily DataFrame with error columns.
        model: Model key ('gfs' or 'ecmwf').

    Returns:
        Mean bias in °C.
    """
    error_col = f"{model}_error"
    return daily_df[error_col].mean()


def compute_monthly_bias(daily_df: pd.DataFrame, model: str) -> pd.Series:
    """Compute mean bias per month.

    Args:
        daily_df: Daily DataFrame with error columns.
        model: Model key ('gfs' or 'ecmwf').

    Returns:
        Series indexed by month with bias values.
    """
    error_col = f"{model}_error"
    df = daily_df.copy()
    df["month"] = df["date"].dt.month
    return df.groupby("month")[error_col].mean()


def compute_bucket_accuracy(daily_df: pd.DataFrame, model: str) -> float:
    """Compute bucket accuracy: % of days where the forecast falls in the correct bucket.

    Args:
        daily_df: Daily DataFrame with forecast and actual columns.
        model: Model key ('gfs' or 'ecmwf').

    Returns:
        Bucket accuracy as a fraction (0-1).
    """
    forecast_col = f"{model}_forecast"
    correct = 0
    total = 0

    for _, row in daily_df.iterrows():
        forecast = row[forecast_col]
        actual = row["actual_temp"]

        if pd.isna(forecast) or pd.isna(actual):
            continue

        month = row["date"].month
        buckets = generate_daily_buckets(month)

        predicted_bucket = assign_bucket(forecast, buckets)
        actual_bucket = assign_bucket(actual, buckets)

        if predicted_bucket is not None and actual_bucket is not None:
            total += 1
            if predicted_bucket["index"] == actual_bucket["index"]:
                correct += 1

    return correct / total if total > 0 else 0.0


def compute_monthly_bucket_accuracy(daily_df: pd.DataFrame, model: str) -> pd.Series:
    """Compute bucket accuracy per month.

    Args:
        daily_df: Daily DataFrame with forecast and actual columns.
        model: Model key ('gfs' or 'ecmwf').

    Returns:
        Series indexed by month with accuracy values (0-1).
    """
    forecast_col = f"{model}_forecast"
    df = daily_df.copy()
    df["month"] = df["date"].dt.month

    results = {}
    for month, group in df.groupby("month"):
        correct = 0
        total = 0
        buckets = generate_daily_buckets(month)

        for _, row in group.iterrows():
            forecast = row[forecast_col]
            actual = row["actual_temp"]
            if pd.isna(forecast) or pd.isna(actual):
                continue

            predicted_bucket = assign_bucket(forecast, buckets)
            actual_bucket = assign_bucket(actual, buckets)
            if predicted_bucket is not None and actual_bucket is not None:
                total += 1
                if predicted_bucket["index"] == actual_bucket["index"]:
                    correct += 1

        results[month] = correct / total if total > 0 else 0.0

    return pd.Series(results)


def compute_all_metrics(daily_df: pd.DataFrame) -> dict:
    """Compute all accuracy metrics for all models.

    Args:
        daily_df: Daily DataFrame with forecast and actual columns.

    Returns:
        Dict with all metrics organized by model.
    """
    metrics = {}

    for model in ["gfs", "ecmwf"]:
        forecast_col = f"{model}_forecast"
        if forecast_col not in daily_df.columns:
            continue

        valid = daily_df[forecast_col].notna().sum()
        metrics[model] = {
            "valid_days": int(valid),
            "mae": compute_mae(daily_df, model),
            "bias": compute_bias(daily_df, model),
            "bucket_accuracy": compute_bucket_accuracy(daily_df, model),
            "monthly_mae": compute_monthly_mae(daily_df, model).to_dict(),
            "monthly_bias": compute_monthly_bias(daily_df, model).to_dict(),
            "monthly_bucket_accuracy": compute_monthly_bucket_accuracy(daily_df, model).to_dict(),
        }

    return metrics
