"""Configuration constants for the London weather backtest."""

from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PLOTS_DIR = PROJECT_ROOT / "output" / "plots"
OUTPUT_REPORTS_DIR = PROJECT_ROOT / "output" / "reports"

# --- Station: London City Airport (EGLC) ---
LATITUDE = 51.5054
LONGITUDE = 0.0553
TIMEZONE = "Europe/London"
STATION_NAME = "London City Airport (EGLC)"

# --- Backtest period ---
DEFAULT_START_DATE = "2025-03-14"
DEFAULT_END_DATE = "2026-03-14"

# --- Weather models ---
MODELS = {
    "gfs": "gfs_seamless",
    "ecmwf": "ecmwf_ifs025",
}

# --- API endpoints ---
PREVIOUS_RUNS_API = "https://previous-runs-api.open-meteo.com/v1/forecast"
HISTORICAL_FORECAST_API = "https://historical-forecast-api.open-meteo.com/v1/forecast"

# --- Lead times (days before target date) ---
LEAD_TIMES = [1, 2]

# --- Temperature variable ---
TEMPERATURE_VARIABLE = "temperature_2m_max"

# --- Bucket configuration ---
DEFAULT_BUCKET_SIZE = 2  # degrees Celsius
DEFAULT_NUM_BUCKETS = 7  # typical number of buckets

# --- London climatology (approximate monthly avg max temp in °C) ---
# Used to center buckets for each day of the year
LONDON_MONTHLY_CLIMATOLOGY = {
    1: 8,    # January
    2: 9,    # February
    3: 11,   # March
    4: 14,   # April
    5: 17,   # May
    6: 20,   # June
    7: 23,   # July
    8: 23,   # August
    9: 20,   # September
    10: 15,  # October
    11: 11,  # November
    12: 8,   # December
}

# --- Simulated market prices ---
PRICE_SCENARIOS = {
    "efficient_market": {
        "consensus_bucket": 0.45,
        "adjacent_bucket": 0.18,
        "far_bucket": 0.03,
    },
    "inefficient_market": {
        "consensus_bucket": 0.30,
        "adjacent_bucket": 0.20,
        "far_bucket": 0.05,
    },
}

# --- Time-decay pricing model ---
# Simulates how Polymarket prices evolve as resolution approaches.
# The idea: the earlier you bet, the cheaper the consensus bucket is
# because the market hasn't fully absorbed the forecast yet.
# Price = base_price at max lead, rising toward fair_price near resolution.
TIME_DECAY_PRICING = {
    # lead_time_hours: consensus_bucket price
    # J-2 evening (~36h before resolution) → market barely moved
    36: {"consensus_bucket": 0.20, "adjacent_bucket": 0.18, "far_bucket": 0.07},
    # J-1 evening (~18h before) → some sharp bettors have entered
    18: {"consensus_bucket": 0.30, "adjacent_bucket": 0.18, "far_bucket": 0.05},
    # J-1 morning / day-of early (~6h before) → market mostly priced in
    6:  {"consensus_bucket": 0.50, "adjacent_bucket": 0.15, "far_bucket": 0.03},
    # Day-of afternoon (~2h before) → almost no alpha left
    2:  {"consensus_bucket": 0.65, "adjacent_bucket": 0.12, "far_bucket": 0.02},
}

# --- Trading parameters ---
STAKE_PER_TRADE = 1.00  # $1 per trade
DEFAULT_PRICE_SCENARIO = "inefficient_market"

# --- Trailing window for Best Model strategy ---
TRAILING_WINDOW_DAYS = 30

# --- API rate limiting ---
API_REQUEST_DELAY = 0.15  # seconds between API calls

# --- Plot styling ---
PLOT_COLORS = {
    "gfs": "#2196F3",
    "ecmwf": "#F44336",
    "actual": "#4CAF50",
}
PLOT_DPI = 150
PLOT_STYLE = "seaborn-v0_8-whitegrid"
