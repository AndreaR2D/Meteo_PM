"""Configuration for the paper trade data collector."""

import os

# --- GCS Storage ---
GCS_BUCKET = os.environ.get("GCS_BUCKET", "bucket-meteo-pm")
GCS_BLOB_NAME = "history.csv"

# --- Cities ---
# Each city: slug name for Polymarket, latitude, longitude, timezone
CITIES = {
    "London": {
        "latitude": 51.5054,
        "longitude": 0.0553,
        "timezone": "Europe/London",
        "slug_name": "london",
    },
    "Tokyo": {
        "latitude": 35.6762,
        "longitude": 139.6503,
        "timezone": "Asia/Tokyo",
        "slug_name": "tokyo",
    },
    "Sao Paulo": {
        "latitude": -23.5505,
        "longitude": -46.6333,
        "timezone": "America/Sao_Paulo",
        "slug_name": "sao-paulo",
    },
}

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
