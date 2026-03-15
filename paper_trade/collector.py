"""Daily data collector for London weather paper trading.

Runs once per day at 19h Paris time (via cron/scheduler).
Each run:
1. Fetches J-2 forecasts (ECMWF + GFS) for TODAY from Open-Meteo
2. Fetches PM resolution for YESTERDAY (winning bucket + actual max temp)
3. Appends/updates rows in history.csv

CSV columns:
    date, ecmwf_j2, gfs_j2, convergence_j2, pm_max_temp

The forecasts are written on day D, the PM resolution is backfilled on day D+1.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

from config import (
    DATA_FILE,
    GAMMA_API,
    LATITUDE,
    LOG_FILE,
    LONGITUDE,
    MODELS,
    MONTH_NAMES,
    PREVIOUS_RUNS_API,
    YEAR_SUFFIX_START_MONTH,
    YEAR_SUFFIX_START_YEAR,
)

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

CSV_COLUMNS = ["date", "ecmwf_j2", "gfs_j2", "convergence_j2", "pm_max_temp"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_slug(target: date) -> str:
    month_name = MONTH_NAMES[target.month]
    slug = f"highest-temperature-in-london-on-{month_name}-{target.day}"
    if (target.year > YEAR_SUFFIX_START_YEAR
            or (target.year == YEAR_SUFFIX_START_YEAR
                and target.month >= YEAR_SUFFIX_START_MONTH)):
        slug += f"-{target.year}"
    return slug


def _read_csv() -> list[dict]:
    """Read existing history.csv into a list of dicts."""
    if not DATA_FILE.exists():
        return []
    with open(DATA_FILE, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _write_csv(rows: list[dict]) -> None:
    """Write rows back to history.csv."""
    with open(DATA_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _find_row(rows: list[dict], target_date: str) -> int | None:
    """Find row index for a given date string (YYYY-MM-DD)."""
    for i, r in enumerate(rows):
        if r["date"] == target_date:
            return i
    return None


# ---------------------------------------------------------------------------
# Open-Meteo: J-2 forecasts
# ---------------------------------------------------------------------------

def fetch_j2_forecasts(target: date) -> dict[str, float | None]:
    """Fetch J-2 forecast (max temp) for target date from both models.

    Uses the Previous Runs API with temperature_2m_previous_day2 (hourly),
    then takes the daily max.

    Returns dict: {"ecmwf": temp_or_None, "gfs": temp_or_None}
    """
    results = {}

    for model_key, model_id in MODELS.items():
        try:
            resp = requests.get(
                PREVIOUS_RUNS_API,
                params={
                    "latitude": LATITUDE,
                    "longitude": LONGITUDE,
                    "start_date": target.isoformat(),
                    "end_date": target.isoformat(),
                    "hourly": "temperature_2m_previous_day2",
                    "models": model_id,
                    "timezone": "UTC",
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            hourly = data.get("hourly", {})
            temps = hourly.get("temperature_2m_previous_day2", [])
            # Filter out None values
            valid = [t for t in temps if t is not None]

            if valid:
                results[model_key] = round(max(valid))
                logger.info(
                    "%s J-2 forecast for %s: %.1f°C (max of %d hourly values)",
                    model_key.upper(), target, results[model_key], len(valid),
                )
            else:
                results[model_key] = None
                logger.warning("%s J-2 forecast for %s: no valid data", model_key.upper(), target)

        except Exception as e:
            logger.error("Failed to fetch %s forecast for %s: %s", model_key.upper(), target, e)
            results[model_key] = None

    return results


# ---------------------------------------------------------------------------
# Polymarket: resolution
# ---------------------------------------------------------------------------

def _parse_winning_temp(label: str) -> float | None:
    """Extract the temperature value from a winning bucket label.

    Examples:
        "12°C" -> 12.0
        "11°C or below" -> 11.0  (the max of the range)
        "19°C or higher" -> 19.0 (the min of the range)
        "45-46°F" -> convert midpoint to °C
    """
    # Normalize degree sign
    label = label.replace("\u00b0", "°")
    # Normalize en-dash
    label = re.sub(r"(\d)\u2013(\d)", r"\1-\2", label)

    # Single °C value: "12°C"
    m = re.match(r"(-?\d+)°C$", label)
    if m:
        return float(m.group(1))

    # "X°C or below"
    m = re.match(r"(-?\d+)°C\s+or\s+below", label)
    if m:
        return float(m.group(1))

    # "X°C or higher"
    m = re.match(r"(-?\d+)°C\s+or\s+higher", label)
    if m:
        return float(m.group(1))

    # °F range: "45-46°F" -> midpoint converted to °C
    m = re.match(r"(\d+)-(\d+)°F", label)
    if m:
        mid_f = (int(m.group(1)) + int(m.group(2))) / 2
        return round((mid_f - 32) * 5 / 9, 1)

    # °F single/bound
    m = re.match(r"(\d+)°F", label)
    if m:
        return round((int(m.group(1)) - 32) * 5 / 9, 1)

    logger.warning("Could not parse temperature from label: %s", label)
    return None


def fetch_pm_resolution(target: date) -> float | None:
    """Fetch the resolved max temperature from Polymarket for target date.

    Finds the winning market (outcomePrices YES=1) and extracts the temp.

    Returns the winning temperature in °C, or None if not resolved yet.
    """
    slug = _build_slug(target)

    try:
        resp = requests.get(GAMMA_API, params={"slug": slug}, timeout=30)
        resp.raise_for_status()
        events = resp.json()
    except Exception as e:
        logger.error("Failed to fetch PM event for %s: %s", target, e)
        return None

    if not events:
        logger.warning("No PM event found for %s (slug=%s)", target, slug)
        return None

    event = events[0]
    markets = event.get("markets", [])

    # Find the winning market via outcomePrices
    winner_label = None
    for mkt in markets:
        outcome_raw = mkt.get("outcomePrices", "")
        if not outcome_raw:
            continue
        try:
            outcomes = json.loads(outcome_raw) if isinstance(outcome_raw, str) else outcome_raw
            yes_price = float(outcomes[0])
            if yes_price >= 0.9:
                winner_label = mkt.get("groupItemTitle", "")
                break
        except (json.JSONDecodeError, ValueError, IndexError):
            continue

    if not winner_label:
        # Market may not have resolved yet
        logger.info("PM market for %s not resolved yet (slug=%s)", target, slug)
        return None

    temp = _parse_winning_temp(winner_label)
    if temp is not None:
        logger.info("PM resolution for %s: %s -> %.1f°C", target, winner_label, temp)
    return temp


# ---------------------------------------------------------------------------
# Main collection logic
# ---------------------------------------------------------------------------

def collect() -> None:
    """Run the daily collection.

    Called at 19h Paris time each day (day D):
    1. Fetch J-2 forecasts for today (D) and write/update row
    2. Fetch PM resolution for yesterday (D-1) and backfill
    """
    today = date.today()
    yesterday = today - timedelta(days=1)

    logger.info("=" * 50)
    logger.info("Collection run: %s", today)
    logger.info("=" * 50)

    rows = _read_csv()

    # --- Step 1: J-2 forecasts for today ---
    logger.info("Fetching J-2 forecasts for %s...", today)
    forecasts = fetch_j2_forecasts(today)

    ecmwf = forecasts.get("ecmwf")
    gfs = forecasts.get("gfs")

    # Convergence: both models predict the same integer °C
    if ecmwf is not None and gfs is not None:
        convergence = "yes" if round(ecmwf) == round(gfs) else "no"
    else:
        convergence = ""

    today_str = today.isoformat()
    idx = _find_row(rows, today_str)

    today_row = {
        "date": today_str,
        "ecmwf_j2": str(int(ecmwf)) if ecmwf is not None else "",
        "gfs_j2": str(int(gfs)) if gfs is not None else "",
        "convergence_j2": convergence,
        "pm_max_temp": "",  # will be filled tomorrow
    }

    if idx is not None:
        # Preserve pm_max_temp if already set
        today_row["pm_max_temp"] = rows[idx].get("pm_max_temp", "")
        rows[idx] = today_row
    else:
        rows.append(today_row)

    logger.info(
        "Today's forecasts: ECMWF=%s°C, GFS=%s°C, convergence=%s",
        ecmwf, gfs, convergence,
    )

    # --- Step 2: PM resolution for yesterday ---
    logger.info("Fetching PM resolution for %s...", yesterday)
    pm_temp = fetch_pm_resolution(yesterday)

    yesterday_str = yesterday.isoformat()
    idx_y = _find_row(rows, yesterday_str)

    if pm_temp is not None:
        pm_int = str(int(pm_temp))
        if idx_y is not None:
            rows[idx_y]["pm_max_temp"] = pm_int
            logger.info("Updated yesterday's PM temp: %s°C", pm_int)
        else:
            # Row for yesterday doesn't exist (first run or missed day)
            rows.append({
                "date": yesterday_str,
                "ecmwf_j2": "",
                "gfs_j2": "",
                "convergence_j2": "",
                "pm_max_temp": pm_int,
            })
            logger.info("Created row for yesterday with PM temp: %s°C", pm_int)
    else:
        logger.info("PM resolution for %s not available yet", yesterday)

    # Sort rows by date and write
    rows.sort(key=lambda r: r["date"])
    _write_csv(rows)
    logger.info("Saved %d rows to %s", len(rows), DATA_FILE)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    collect()
