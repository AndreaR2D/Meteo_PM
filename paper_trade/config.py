"""Configuration for the paper trade data collector."""

from pathlib import Path

# --- Paths ---
PROJECT_DIR = Path(__file__).parent
DATA_FILE = PROJECT_DIR / "history.csv"
LOG_FILE = PROJECT_DIR / "collector.log"

# --- Station: London City Airport (EGLC) ---
LATITUDE = 51.5054
LONGITUDE = 0.0553
TIMEZONE = "Europe/London"

# --- APIs ---
PREVIOUS_RUNS_API = "https://previous-runs-api.open-meteo.com/v1/forecast"
FORECAST_API = "https://api.open-meteo.com/v1/forecast"
GAMMA_API = "https://gamma-api.polymarket.com/events"

# --- Weather models ---
MODELS = {
    "gfs": "gfs_seamless",
    "ecmwf": "ecmwf_ifs025",
}

# --- Slug config ---
# Dates >= this use -YYYY suffix in the slug
YEAR_SUFFIX_START_MONTH = 3  # March
YEAR_SUFFIX_START_YEAR = 2026

MONTH_NAMES = {
    1: "january", 2: "february", 3: "march", 4: "april",
    5: "may", 6: "june", 7: "july", 8: "august",
    9: "september", 10: "october", 11: "november", 12: "december",
}
