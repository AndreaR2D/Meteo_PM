"""Daily data collector for multi-city weather paper trading.

Single daily run at 07:00 UTC (via cron/scheduler).
Each run, for each city (London, Tokyo, Sao Paulo):
1. Fetches J-2, J-1, J forecasts for TODAY from Open-Meteo
2. Fetches PM resolution for YESTERDAY (backfill pm_max_temp)
3. Appends/updates rows in history.csv

CSV columns:
    ville, date, ecmwf_j2, gfs_j2, convergence_j2,
    ecmwf_j1, gfs_j1, convergence_j1,
    ecmwf_j, gfs_j, convergence_j,
    pm_max_temp
"""

from __future__ import annotations

import csv
import io
import json
import logging
import re
import sys
from datetime import date, timedelta

import requests
from google.cloud import storage

from config import (
    CITIES,
    FORECAST_API,
    GAMMA_API,
    GCS_BLOB_NAME,
    GCS_BUCKET,
    MODELS,
    MONTH_NAMES,
    PREVIOUS_RUNS_API,
    YEAR_SUFFIX_START_MONTH,
    YEAR_SUFFIX_START_YEAR,
)

# --- Logging setup (stdout only for Cloud Run) ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# --- GCS client (reused across invocations) ---
_gcs_client: storage.Client | None = None


def _get_bucket() -> storage.Bucket:
    global _gcs_client
    if _gcs_client is None:
        _gcs_client = storage.Client()
    return _gcs_client.bucket(GCS_BUCKET)

CSV_COLUMNS = [
    "ville", "date",
    "ecmwf_j2", "gfs_j2", "convergence_j2",
    "ecmwf_j1", "gfs_j1", "convergence_j1",
    "ecmwf_j", "gfs_j", "convergence_j",
    "pm_max_temp",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_slug(target: date, slug_name: str) -> str:
    month_name = MONTH_NAMES[target.month]
    slug = f"highest-temperature-in-{slug_name}-on-{month_name}-{target.day}"
    if (target.year > YEAR_SUFFIX_START_YEAR
            or (target.year == YEAR_SUFFIX_START_YEAR
                and target.month >= YEAR_SUFFIX_START_MONTH)):
        slug += f"-{target.year}"
    return slug


def _read_csv() -> list[dict]:
    """Read existing history.csv from GCS bucket."""
    blob = _get_bucket().blob(GCS_BLOB_NAME)
    if not blob.exists():
        logger.info("No existing %s in gs://%s, starting fresh", GCS_BLOB_NAME, GCS_BUCKET)
        return []
    content = blob.download_as_text(encoding="utf-8")
    reader = csv.DictReader(io.StringIO(content))
    rows = []
    for row in reader:
        row.pop(None, None)
        for col in CSV_COLUMNS:
            row.setdefault(col, "")
        rows.append(row)
    return rows


def _write_csv(rows: list[dict]) -> None:
    """Write rows to history.csv in GCS bucket (overwrites)."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    blob = _get_bucket().blob(GCS_BLOB_NAME)
    blob.upload_from_string(buf.getvalue(), content_type="text/csv")
    logger.info("Uploaded %s to gs://%s/%s", GCS_BLOB_NAME, GCS_BUCKET, GCS_BLOB_NAME)


def _find_row(rows: list[dict], ville: str, target_date: str) -> int | None:
    """Find row index for a given city + date."""
    for i, r in enumerate(rows):
        if r["ville"] == ville and r["date"] == target_date:
            return i
    return None


# ---------------------------------------------------------------------------
# Open-Meteo: J-2 forecasts
# ---------------------------------------------------------------------------

def fetch_j2_forecasts(target: date, city: dict) -> dict[str, float | None]:
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
                    "latitude": city["latitude"],
                    "longitude": city["longitude"],
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
# Open-Meteo: J-1 forecasts
# ---------------------------------------------------------------------------

def fetch_j1_forecasts(target: date, city: dict) -> dict[str, float | None]:
    """Fetch J-1 forecast (max temp) for target date from both models.

    Uses the Previous Runs API with temperature_2m_previous_day1 (hourly),
    then takes the daily max.

    Returns dict: {"ecmwf": temp_or_None, "gfs": temp_or_None}
    """
    results = {}

    for model_key, model_id in MODELS.items():
        try:
            resp = requests.get(
                PREVIOUS_RUNS_API,
                params={
                    "latitude": city["latitude"],
                    "longitude": city["longitude"],
                    "start_date": target.isoformat(),
                    "end_date": target.isoformat(),
                    "hourly": "temperature_2m_previous_day1",
                    "models": model_id,
                    "timezone": "UTC",
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            hourly = data.get("hourly", {})
            temps = hourly.get("temperature_2m_previous_day1", [])
            valid = [t for t in temps if t is not None]

            if valid:
                results[model_key] = round(max(valid))
                logger.info(
                    "%s J-1 forecast for %s: %.1f°C (max of %d hourly values)",
                    model_key.upper(), target, results[model_key], len(valid),
                )
            else:
                results[model_key] = None
                logger.warning("%s J-1 forecast for %s: no valid data", model_key.upper(), target)

        except Exception as e:
            logger.error("Failed to fetch %s J-1 forecast for %s: %s", model_key.upper(), target, e)
            results[model_key] = None

    return results


# ---------------------------------------------------------------------------
# Open-Meteo: J forecasts (same-day, morning run)
# ---------------------------------------------------------------------------

def fetch_j_forecasts(target: date, city: dict) -> dict[str, float | None]:
    """Fetch same-day forecast (max temp) for target date from both models.

    Uses the standard Forecast API with daily temperature_2m_max.
    Intended to run at 07:00 UTC to capture the latest morning model run.

    Returns dict: {"ecmwf": temp_or_None, "gfs": temp_or_None}
    """
    results = {}

    for model_key, model_id in MODELS.items():
        try:
            resp = requests.get(
                FORECAST_API,
                params={
                    "latitude": city["latitude"],
                    "longitude": city["longitude"],
                    "start_date": target.isoformat(),
                    "end_date": target.isoformat(),
                    "daily": "temperature_2m_max",
                    "models": model_id,
                    "timezone": "UTC",
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            daily = data.get("daily", {})
            temps = daily.get("temperature_2m_max", [])
            valid = [t for t in temps if t is not None]

            if valid:
                results[model_key] = round(max(valid))
                logger.info(
                    "%s J forecast for %s: %.1f°C",
                    model_key.upper(), target, results[model_key],
                )
            else:
                results[model_key] = None
                logger.warning("%s J forecast for %s: no valid data", model_key.upper(), target)

        except Exception as e:
            logger.error("Failed to fetch %s J forecast for %s: %s", model_key.upper(), target, e)
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


def fetch_pm_resolution(target: date, slug_name: str) -> float | None:
    """Fetch the resolved max temperature from Polymarket for target date.

    Finds the winning market (outcomePrices YES=1) and extracts the temp.

    Returns the winning temperature in °C, or None if not resolved yet.
    """
    slug = _build_slug(target, slug_name)

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

def _convergence(ecmwf: float | None, gfs: float | None) -> str:
    """Return 'yes'/'no'/'' for model convergence."""
    if ecmwf is not None and gfs is not None:
        return "yes" if round(ecmwf) == round(gfs) else "no"
    return ""


def _fmt(val: float | None) -> str:
    return str(int(val)) if val is not None else ""


def _ensure_row(rows: list[dict], ville: str, date_str: str) -> int:
    """Return index of existing row for ville+date, or append a blank row."""
    idx = _find_row(rows, ville, date_str)
    if idx is not None:
        return idx
    blank = {col: "" for col in CSV_COLUMNS}
    blank["ville"] = ville
    blank["date"] = date_str
    rows.append(blank)
    return len(rows) - 1


def collect() -> None:
    """Daily run (07:00 UTC).

    Called each day (day D), for each city:
    1. Fetch J-2, J-1, J forecasts for today (D)
    2. Fetch PM resolution for yesterday (D-1) and backfill
    """
    today = date.today()
    yesterday = today - timedelta(days=1)

    logger.info("=" * 50)
    logger.info("Collection run: %s", today)
    logger.info("=" * 50)

    rows = _read_csv()

    for ville, city in CITIES.items():
        logger.info("-" * 40)
        logger.info("City: %s", ville)
        logger.info("-" * 40)

        today_str = today.isoformat()
        idx = _ensure_row(rows, ville, today_str)

        # --- Step 1: J-2 forecasts for today ---
        logger.info("Fetching J-2 forecasts for %s / %s...", ville, today)
        fc2 = fetch_j2_forecasts(today, city)
        ecmwf2, gfs2 = fc2.get("ecmwf"), fc2.get("gfs")

        rows[idx]["ecmwf_j2"] = _fmt(ecmwf2)
        rows[idx]["gfs_j2"] = _fmt(gfs2)
        rows[idx]["convergence_j2"] = _convergence(ecmwf2, gfs2)

        logger.info(
            "J-2 forecasts: ECMWF=%s°C, GFS=%s°C, convergence=%s",
            ecmwf2, gfs2, rows[idx]["convergence_j2"],
        )

        # --- Step 2: J-1 forecasts for today ---
        logger.info("Fetching J-1 forecasts for %s / %s...", ville, today)
        fc1 = fetch_j1_forecasts(today, city)
        ecmwf1, gfs1 = fc1.get("ecmwf"), fc1.get("gfs")

        rows[idx]["ecmwf_j1"] = _fmt(ecmwf1)
        rows[idx]["gfs_j1"] = _fmt(gfs1)
        rows[idx]["convergence_j1"] = _convergence(ecmwf1, gfs1)

        logger.info(
            "J-1 forecasts: ECMWF=%s°C, GFS=%s°C, convergence=%s",
            ecmwf1, gfs1, rows[idx]["convergence_j1"],
        )

        # --- Step 3: J forecasts for today ---
        logger.info("Fetching J forecasts for %s / %s...", ville, today)
        fc0 = fetch_j_forecasts(today, city)
        ecmwf0, gfs0 = fc0.get("ecmwf"), fc0.get("gfs")

        rows[idx]["ecmwf_j"] = _fmt(ecmwf0)
        rows[idx]["gfs_j"] = _fmt(gfs0)
        rows[idx]["convergence_j"] = _convergence(ecmwf0, gfs0)

        logger.info(
            "J forecasts: ECMWF=%s°C, GFS=%s°C, convergence=%s",
            ecmwf0, gfs0, rows[idx]["convergence_j"],
        )

        # --- Step 4: PM resolution for yesterday ---
        logger.info("Fetching PM resolution for %s / %s...", ville, yesterday)
        pm_temp = fetch_pm_resolution(yesterday, city["slug_name"])

        yesterday_str = yesterday.isoformat()
        idx_y = _ensure_row(rows, ville, yesterday_str)

        if pm_temp is not None:
            rows[idx_y]["pm_max_temp"] = _fmt(pm_temp)
            logger.info("Updated yesterday's PM temp for %s: %s°C", ville, _fmt(pm_temp))
        else:
            logger.info("PM resolution for %s / %s not available yet", ville, yesterday)

    # Sort rows by ville then date and write
    rows.sort(key=lambda r: (r["ville"], r["date"]))
    _write_csv(rows)
    logger.info("Saved %d rows to gs://%s/%s", len(rows), GCS_BUCKET, GCS_BLOB_NAME)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    collect()
