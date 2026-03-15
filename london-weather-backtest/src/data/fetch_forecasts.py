"""Fetch archived model forecasts from Open-Meteo Previous Runs API.

The Previous Runs API uses hourly data with a special variable naming convention:
- temperature_2m = current model run forecast
- temperature_2m_previous_day1 = forecast made 1 day ago
- temperature_2m_previous_day2 = forecast made 2 days ago

We fetch hourly data and aggregate to daily max to get temperature_2m_max equivalent.
"""

import logging
import time
from pathlib import Path

import pandas as pd
import requests

from config import (
    API_REQUEST_DELAY,
    DATA_RAW_DIR,
    LATITUDE,
    LONGITUDE,
    MODELS,
    PREVIOUS_RUNS_API,
    TIMEZONE,
)

logger = logging.getLogger(__name__)


def _cache_path(model_key: str, lead_time: int) -> Path:
    """Return the cache file path for a given model and lead time."""
    return DATA_RAW_DIR / f"forecast_{model_key}_lead{lead_time}.csv"


def fetch_model_forecast(
    model_key: str,
    model_id: str,
    start_date: str,
    end_date: str,
    lead_time: int = 1,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch archived forecasts for a single model and lead time.

    Uses the Previous Runs API with hourly temperature_2m_previous_dayN,
    then aggregates to daily max temperature.

    Args:
        model_key: Short name for the model (e.g. 'gfs', 'ecmwf').
        model_id: Open-Meteo model identifier (e.g. 'gfs_seamless').
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date in YYYY-MM-DD format.
        lead_time: Number of days before the target date (1 = J-1).
        refresh: If True, re-download even if cached.

    Returns:
        DataFrame with columns ['date', '{model_key}_forecast'].
    """
    cache = _cache_path(model_key, lead_time)

    if cache.exists() and not refresh:
        logger.info("Loading cached forecast: %s", cache.name)
        df = pd.read_csv(cache, parse_dates=["date"])
        return df

    logger.info(
        "Fetching %s forecasts (lead=%d) from %s to %s ...",
        model_key.upper(), lead_time, start_date, end_date,
    )

    # The variable name encodes the lead time
    hourly_var = f"temperature_2m_previous_day{lead_time}"

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": hourly_var,
        "timezone": TIMEZONE,
        "models": model_id,
    }

    time.sleep(API_REQUEST_DELAY)
    response = requests.get(PREVIOUS_RUNS_API, params=params, timeout=120)
    response.raise_for_status()
    data = response.json()

    if "hourly" not in data or hourly_var not in data["hourly"]:
        logger.warning("No hourly data returned for %s (lead=%d)", model_key, lead_time)
        return pd.DataFrame(columns=["date", f"{model_key}_forecast"])

    hourly = data["hourly"]
    hourly_df = pd.DataFrame({
        "datetime": pd.to_datetime(hourly["time"]),
        "temp": hourly[hourly_var],
    })

    # Aggregate hourly to daily max
    hourly_df["date"] = hourly_df["datetime"].dt.date
    daily = hourly_df.groupby("date")["temp"].max().reset_index()
    daily.columns = ["date", f"{model_key}_forecast"]
    daily["date"] = pd.to_datetime(daily["date"])

    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_csv(cache, index=False)
    logger.info("Saved %d rows to %s", len(daily), cache.name)

    return daily


def fetch_all_forecasts(
    start_date: str,
    end_date: str,
    lead_time: int = 1,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch forecasts for all configured models and merge into one DataFrame.

    Args:
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date in YYYY-MM-DD format.
        lead_time: Number of days before the target date.
        refresh: If True, re-download even if cached.

    Returns:
        DataFrame with columns ['date', 'gfs_forecast', 'ecmwf_forecast'].
    """
    merged: pd.DataFrame | None = None

    for model_key, model_id in MODELS.items():
        df = fetch_model_forecast(
            model_key, model_id, start_date, end_date, lead_time, refresh
        )
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on="date", how="outer")

    if merged is None:
        return pd.DataFrame(columns=["date"])

    merged = merged.sort_values("date").reset_index(drop=True)
    return merged
