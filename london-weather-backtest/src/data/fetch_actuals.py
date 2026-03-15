"""Fetch actual observed temperatures from Open-Meteo APIs.

Two sources available:
- Archive API: Actual station observations (closest to Weather Underground).
- Historical Forecast API (best_match): Model reanalysis data (less accurate proxy).

The Archive API is used by default as it better matches Polymarket's resolution
source (Weather Underground EGLC station).
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
    TEMPERATURE_VARIABLE,
    TIMEZONE,
)

logger = logging.getLogger(__name__)

ARCHIVE_API = "https://archive-api.open-meteo.com/v1/archive"
HISTORICAL_FORECAST_API = "https://historical-forecast-api.open-meteo.com/v1/forecast"


def fetch_actuals(
    start_date: str,
    end_date: str,
    refresh: bool = False,
    source: str = "archive",
) -> pd.DataFrame:
    """Fetch actual observed max temperatures for London City Airport.

    Args:
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date in YYYY-MM-DD format.
        refresh: If True, re-download even if cached.
        source: 'archive' for station observations (recommended),
                'best_match' for model reanalysis (legacy).

    Returns:
        DataFrame with columns ['date', 'actual_temp'].
    """
    cache_name = f"actuals_{source}.csv"
    cache = DATA_RAW_DIR / cache_name

    if cache.exists() and not refresh:
        logger.info("Loading cached actuals (%s): %s", source, cache.name)
        df = pd.read_csv(cache, parse_dates=["date"])
        return df

    logger.info(
        "Fetching actual temperatures (%s) from %s to %s ...",
        source, start_date, end_date,
    )

    if source == "archive":
        params = {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "start_date": start_date,
            "end_date": end_date,
            "daily": TEMPERATURE_VARIABLE,
            "timezone": TIMEZONE,
        }
        api_url = ARCHIVE_API
    else:
        params = {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "start_date": start_date,
            "end_date": end_date,
            "daily": TEMPERATURE_VARIABLE,
            "timezone": TIMEZONE,
            "models": "best_match",
        }
        api_url = HISTORICAL_FORECAST_API

    time.sleep(API_REQUEST_DELAY)
    response = requests.get(api_url, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()

    if "daily" not in data:
        logger.warning("No daily data returned for actuals (%s)", source)
        return pd.DataFrame(columns=["date", "actual_temp"])

    daily = data["daily"]
    df = pd.DataFrame({
        "date": pd.to_datetime(daily["time"]),
        "actual_temp": daily[TEMPERATURE_VARIABLE],
    })

    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    logger.info("Saved %d rows to %s", len(df), cache.name)

    return df
