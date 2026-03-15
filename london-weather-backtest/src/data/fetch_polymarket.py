"""Fetch historical Polymarket prices for London weather markets.

Uses:
- Gamma API to discover events/markets and their token IDs + bucket structure
- CLOB API to fetch 12h-granularity price history for each market

Slug patterns:
- March 14 2025 → Feb 28 2026: highest-temperature-in-london-on-{month}-{day}
- March 1 2026+: highest-temperature-in-london-on-{month}-{day}-2026

Bucket transitions:
- Before ~Dec 12 2025: Fahrenheit, 2°F ranges, 7 buckets
- After ~Dec 15 2025: Celsius, 1°C ranges, 7-9 buckets
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from config import API_REQUEST_DELAY, DATA_RAW_DIR

logger = logging.getLogger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com/events"
CLOB_PRICES_API = "https://clob.polymarket.com/prices-history"

# Cache directory for raw PM data
PM_CACHE_DIR = DATA_RAW_DIR / "polymarket"

# Month name mapping for slug generation
MONTH_NAMES = {
    1: "january", 2: "february", 3: "march", 4: "april",
    5: "may", 6: "june", 7: "july", 8: "august",
    9: "september", 10: "october", 11: "november", 12: "december",
}

# Date when slug format changed from no-year to -2026
YEAR_SUFFIX_START = date(2026, 3, 1)

# Approximate transition from °F to °C buckets
CELSIUS_TRANSITION = date(2025, 12, 12)


def _build_slug(target_date: date) -> str:
    """Build the Polymarket event slug for a given date.

    Args:
        target_date: The date of the weather market.

    Returns:
        The event slug string.
    """
    month_name = MONTH_NAMES[target_date.month]
    day = target_date.day
    base = f"highest-temperature-in-london-on-{month_name}-{day}"
    if target_date >= YEAR_SUFFIX_START:
        base += f"-{target_date.year}"
    return base


def _parse_bucket_label(label: str) -> dict:
    """Parse a Polymarket bucket label into structured data.

    Handles both °F and °C formats, including:
    - "42°F or below", "43-44°F", "53°F or higher"
    - "67\u201368°F" (en-dash U+2013)
    - "7°C or below", "8°C", "13°C or higher"
    - "-2°C or below", "-1°C"

    Returns dict with keys: label, unit, low, high, is_lower_bound, is_upper_bound
    """
    # Normalize: replace en-dash (U+2013) and em-dash (U+2014) with ASCII hyphen
    # but only between digits (to preserve negative signs)
    normalized = re.sub(r"(\d)\u2013(\d)", r"\1-\2", label)
    normalized = re.sub(r"(\d)\u2014(\d)", r"\1-\2", normalized)

    # Also normalize degree sign variations
    normalized = normalized.replace("\u00b0", "°")

    result = {"label": label, "unit": "C", "low": None, "high": None,
              "is_lower_bound": False, "is_upper_bound": False}

    if "°F" in normalized or "°F" in label:
        result["unit"] = "F"
    elif "°C" in normalized or "°C" in label:
        result["unit"] = "C"

    # "X or below" / "X or lower" (handles negative temps)
    m = re.match(r"(-?\d+)°[FC]\s+or\s+below", normalized, re.IGNORECASE)
    if m:
        result["high"] = int(m.group(1))
        result["is_lower_bound"] = True
        return result

    # "X or higher" / "X or above"
    m = re.match(r"(-?\d+)°[FC]\s+or\s+(higher|above)", normalized, re.IGNORECASE)
    if m:
        result["low"] = int(m.group(1))
        result["is_upper_bound"] = True
        return result

    # "X-Y°F" range (X and Y are positive, dash separates them)
    m = re.match(r"(\d+)-(\d+)°[FC]", normalized)
    if m:
        result["low"] = int(m.group(1))
        result["high"] = int(m.group(2))
        return result

    # Single value "X°C" or "-X°C"
    m = re.match(r"(-?\d+)°[FC]$", normalized)
    if m:
        val = int(m.group(1))
        result["low"] = val
        result["high"] = val
        return result

    logger.warning("Could not parse bucket label: %s (normalized: %s)", label, normalized)
    return result


def _fahrenheit_to_celsius(temp_f: float) -> float:
    """Convert Fahrenheit to Celsius."""
    return (temp_f - 32) * 5 / 9


def _bucket_contains_temp_c(bucket: dict, temp_c: float) -> bool:
    """Check if a temperature (in °C) falls within a bucket.

    Handles unit conversion for °F buckets.
    The temp_c is rounded to nearest integer for °C buckets,
    or converted and checked against °F ranges.
    """
    if bucket["unit"] == "C":
        temp_rounded = round(temp_c)
        if bucket["is_lower_bound"]:
            return temp_rounded <= bucket["high"]
        if bucket["is_upper_bound"]:
            return temp_rounded >= bucket["low"]
        return bucket["low"] <= temp_rounded <= bucket["high"]
    else:
        # Convert °C to °F for comparison
        temp_f = temp_c * 9 / 5 + 32
        temp_f_rounded = round(temp_f)
        if bucket["is_lower_bound"]:
            return temp_f_rounded <= bucket["high"]
        if bucket["is_upper_bound"]:
            return temp_f_rounded >= bucket["low"]
        return bucket["low"] <= temp_f_rounded <= bucket["high"]


def fetch_event(target_date: date, refresh: bool = False) -> dict | None:
    """Fetch event data from Gamma API for a given date.

    Args:
        target_date: The date of the weather market.
        refresh: Re-fetch even if cached.

    Returns:
        Dict with event data including markets, or None if not found.
    """
    PM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = PM_CACHE_DIR / f"event_{target_date.isoformat()}.json"

    if cache_path.exists() and not refresh:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if data else None

    slug = _build_slug(target_date)
    time.sleep(API_REQUEST_DELAY)

    try:
        resp = requests.get(GAMMA_API, params={"slug": slug}, timeout=30)
        resp.raise_for_status()
        events = resp.json()
    except (requests.RequestException, json.JSONDecodeError) as e:
        logger.warning("Failed to fetch event for %s (slug=%s): %s", target_date, slug, e)
        # Cache the failure
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(None, f)
        return None

    if not events:
        logger.debug("No event found for %s (slug=%s)", target_date, slug)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(None, f)
        return None

    event = events[0]

    # Cache
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(event, f)

    return event


def fetch_market_prices(token_id: str, refresh: bool = False) -> list[dict]:
    """Fetch price history for a single market token.

    Args:
        token_id: The CLOB token ID (YES token).
        refresh: Re-fetch even if cached.

    Returns:
        List of {t: unix_timestamp, p: price} dicts.
    """
    PM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # Use short hash for filename since token IDs are very long
    token_hash = str(hash(token_id) % (10**12))
    cache_path = PM_CACHE_DIR / f"prices_{token_hash}.json"

    if cache_path.exists() and not refresh:
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    time.sleep(API_REQUEST_DELAY)

    try:
        resp = requests.get(
            CLOB_PRICES_API,
            params={"market": token_id, "interval": "max", "fidelity": 720},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, json.JSONDecodeError) as e:
        logger.warning("Failed to fetch prices for token %s...: %s", token_id[:20], e)
        return []

    history = data.get("history", [])

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(history, f)

    return history


def _extract_market_data(event: dict, target_date: date) -> list[dict]:
    """Extract structured market/bucket data from a Gamma API event.

    Returns list of dicts with: bucket_label, bucket_parsed, yes_token_id,
    last_trade_price, outcome_prices, prices_history.
    """
    markets = event.get("markets", [])
    result = []

    for market in markets:
        label = market.get("groupItemTitle", "")
        if not label:
            continue

        token_ids_raw = market.get("clobTokenIds", "[]")
        token_ids = json.loads(token_ids_raw) if isinstance(token_ids_raw, str) else token_ids_raw
        if not token_ids:
            continue

        yes_token = token_ids[0]
        bucket = _parse_bucket_label(label)

        # Parse outcomePrices to determine resolved YES price
        outcome_raw = market.get("outcomePrices", "")
        resolved_yes_price = None
        if outcome_raw:
            try:
                outcomes = json.loads(outcome_raw) if isinstance(outcome_raw, str) else outcome_raw
                if outcomes and len(outcomes) >= 1:
                    resolved_yes_price = float(outcomes[0])
            except (json.JSONDecodeError, ValueError):
                pass

        result.append({
            "date": target_date.isoformat(),
            "bucket_label": label,
            "bucket_parsed": bucket,
            "yes_token_id": yes_token,
            "last_trade_price": market.get("lastTradePrice"),
            "resolved_yes_price": resolved_yes_price,
            "outcome_prices": market.get("outcomePrices"),
        })

    return result


def fetch_day_prices(
    target_date: date,
    refresh: bool = False,
) -> dict | None:
    """Fetch all market data and price histories for a single day.

    Args:
        target_date: The date of the weather market.
        refresh: Re-fetch even if cached.

    Returns:
        Dict with keys: date, event_title, buckets (list of bucket data
        with price histories), or None if no data.
    """
    event = fetch_event(target_date, refresh=refresh)
    if event is None:
        return None

    markets_data = _extract_market_data(event, target_date)
    if not markets_data:
        return None

    # Fetch price histories for each market
    for mkt in markets_data:
        history = fetch_market_prices(mkt["yes_token_id"], refresh=refresh)
        mkt["prices_history"] = history

    return {
        "date": target_date.isoformat(),
        "event_title": event.get("title", ""),
        "buckets": markets_data,
    }


def _get_consensus_bucket_price(
    day_data: dict,
    forecast_temp_c: float,
    target_hours_before: float = 18.0,
) -> tuple[float | None, str | None]:
    """Find the price of the bucket matching the forecast temperature.

    Looks at the price history to find the price closest to target_hours_before
    the resolution time (end of target date).

    Args:
        day_data: Output from fetch_day_prices().
        forecast_temp_c: The forecasted temperature in °C.
        target_hours_before: How many hours before resolution to sample the price.

    Returns:
        Tuple of (price, bucket_label) or (None, None) if no match.
    """
    if day_data is None:
        return None, None

    target_date = date.fromisoformat(day_data["date"])
    # Resolution is roughly end of day UTC
    resolution_dt = datetime(target_date.year, target_date.month, target_date.day, 23, 59)

    # Find the bucket that the forecast falls into
    matching_bucket = None
    for bucket_data in day_data["buckets"]:
        parsed = bucket_data["bucket_parsed"]
        if _bucket_contains_temp_c(parsed, forecast_temp_c):
            matching_bucket = bucket_data
            break

    if matching_bucket is None:
        return None, None

    # Find the price closest to target_hours_before resolution
    history = matching_bucket.get("prices_history", [])
    if not history:
        # Fall back to last_trade_price if no history
        return matching_bucket.get("last_trade_price"), matching_bucket["bucket_label"]

    target_ts = (resolution_dt - timedelta(hours=target_hours_before)).timestamp()

    # Find closest point
    best = min(history, key=lambda h: abs(h["t"] - target_ts))
    return best["p"], matching_bucket["bucket_label"]


def fetch_all_pm_prices(
    start_date: str,
    end_date: str,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch Polymarket prices for all dates in the backtest period.

    Args:
        start_date: Start date YYYY-MM-DD.
        end_date: End date YYYY-MM-DD.
        refresh: Re-fetch even if cached.

    Returns:
        DataFrame with columns: date, event_found, num_buckets, unit,
        and per-bucket price snapshots at various lead times.
    """
    cache_path = DATA_RAW_DIR / "polymarket_prices.csv"
    if cache_path.exists() and not refresh:
        logger.info("Loading cached Polymarket prices: %s", cache_path.name)
        return pd.read_csv(cache_path, parse_dates=["date"])

    d_start = date.fromisoformat(start_date)
    d_end = date.fromisoformat(end_date)
    total_days = (d_end - d_start).days + 1

    rows = []
    for i in range(total_days):
        current = d_start + timedelta(days=i)

        if i % 30 == 0:
            logger.info("Fetching PM data: day %d/%d (%s)...", i + 1, total_days, current)

        day_data = fetch_day_prices(current, refresh=refresh)

        row = {
            "date": pd.Timestamp(current),
            "event_found": day_data is not None,
            "num_buckets": 0,
            "unit": None,
            "buckets_json": None,
        }

        if day_data is not None:
            buckets = day_data["buckets"]
            row["num_buckets"] = len(buckets)
            if buckets:
                row["unit"] = buckets[0]["bucket_parsed"]["unit"]

            # Store full bucket data as JSON for later analysis
            # Strip the parsed dict and large token IDs for compactness
            compact_buckets = []
            for b in buckets:
                compact = {
                    "label": b["bucket_label"],
                    "unit": b["bucket_parsed"]["unit"],
                    "low": b["bucket_parsed"]["low"],
                    "high": b["bucket_parsed"]["high"],
                    "is_lower": b["bucket_parsed"]["is_lower_bound"],
                    "is_upper": b["bucket_parsed"]["is_upper_bound"],
                    "last_price": b.get("last_trade_price"),
                    "resolved_yes": b.get("resolved_yes_price"),
                    "prices": b.get("prices_history", []),
                    "yes_token": b["yes_token_id"],
                }
                compact_buckets.append(compact)
            row["buckets_json"] = json.dumps(compact_buckets)

        rows.append(row)

    df = pd.DataFrame(rows)

    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    logger.info("Saved %d days of PM data to %s", len(df), cache_path.name)

    return df


def get_real_price_for_trade(
    pm_df: pd.DataFrame,
    trade_date: date,
    forecast_temp_c: float,
    lead_hours: float = 18.0,
) -> float | None:
    """Look up the real Polymarket price for a given trade.

    Args:
        pm_df: DataFrame from fetch_all_pm_prices().
        trade_date: The date of the trade.
        forecast_temp_c: The forecast temperature in °C.
        lead_hours: Hours before resolution when we'd buy.

    Returns:
        The real market price, or None if not available.
    """
    row = pm_df[pm_df["date"].dt.date == trade_date]
    if row.empty or not row.iloc[0]["event_found"]:
        return None

    buckets_json = row.iloc[0]["buckets_json"]
    if pd.isna(buckets_json):
        return None

    buckets = json.loads(buckets_json)

    # Find matching bucket
    matching = None
    for b in buckets:
        bucket_parsed = {
            "unit": b["unit"], "low": b["low"], "high": b["high"],
            "is_lower_bound": b["is_lower"], "is_upper_bound": b["is_upper"],
        }
        if _bucket_contains_temp_c(bucket_parsed, forecast_temp_c):
            matching = b
            break

    if matching is None:
        return None

    prices = matching.get("prices", [])
    if not prices:
        return matching.get("last_price")

    # Resolution is roughly end of day
    resolution_dt = datetime(trade_date.year, trade_date.month, trade_date.day, 23, 59)
    target_ts = (resolution_dt - timedelta(hours=lead_hours)).timestamp()

    best = min(prices, key=lambda h: abs(h["t"] - target_ts))
    return best["p"]


def get_actual_winning_bucket(
    pm_df: pd.DataFrame,
    trade_date: date,
) -> str | None:
    """Determine which bucket won based on final Polymarket prices.

    The winning bucket is the one with the highest final price (closest to 1.0).

    Args:
        pm_df: DataFrame from fetch_all_pm_prices().
        trade_date: The date to check.

    Returns:
        The label of the winning bucket, or None.
    """
    row = pm_df[pm_df["date"].dt.date == trade_date]
    if row.empty or not row.iloc[0]["event_found"]:
        return None

    buckets_json = row.iloc[0]["buckets_json"]
    if pd.isna(buckets_json):
        return None

    buckets = json.loads(buckets_json)

    # The winning bucket has the highest last_price (should be ~1.0)
    best = max(buckets, key=lambda b: b.get("last_price", 0) or 0)
    if (best.get("last_price") or 0) > 0.5:
        return best["label"]

    return None
