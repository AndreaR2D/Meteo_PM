"""V2 Backtest Analysis — Real Polymarket Prices.

Compares the simulated backtest results with actual Polymarket market prices
to determine if the convergence strategy has real alpha.
"""

import json
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    OUTPUT_PLOTS_DIR,
    PLOT_COLORS,
    PLOT_DPI,
    PLOT_STYLE,
)
from src.data.fetch_polymarket import (
    _bucket_contains_temp_c,
    _parse_bucket_label,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the daily forecasts and Polymarket prices data."""
    daily_path = DATA_PROCESSED_DIR / "forecasts_daily.csv"
    pm_path = DATA_RAW_DIR / "polymarket_prices.csv"

    daily = pd.read_csv(daily_path, parse_dates=["date"])
    pm = pd.read_csv(pm_path, parse_dates=["date"])

    return daily, pm


def _find_bucket_price(
    buckets: list[dict],
    temp_c: float,
    target_ts: float,
) -> tuple[float | None, str | None]:
    """Find the price of the bucket matching temp_c at the closest time to target_ts."""
    for b in buckets:
        parsed = {
            "unit": b["unit"], "low": b["low"], "high": b["high"],
            "is_lower_bound": b["is_lower"], "is_upper_bound": b["is_upper"],
        }
        if _bucket_contains_temp_c(parsed, temp_c):
            prices = b.get("prices", [])
            if not prices:
                return b.get("last_price"), b["label"]
            best = min(prices, key=lambda h: abs(h["t"] - target_ts))
            return best["p"], b["label"]
    return None, None


def _find_winning_bucket(buckets: list[dict]) -> str | None:
    """Find the bucket that won.

    Uses resolved_yes_price (from outcomePrices) as primary indicator.
    Falls back to last_price or final price history point.
    """
    # Primary: use resolved_yes_price (outcomePrices[0])
    for b in buckets:
        resolved = b.get("resolved_yes")
        if resolved is not None and resolved >= 0.9:
            return b["label"]

    # Fallback: check last price history point
    best_final = None
    best_final_price = 0
    for b in buckets:
        prices = b.get("prices", [])
        if prices:
            final_p = prices[-1]["p"]
            if final_p > best_final_price:
                best_final_price = final_p
                best_final = b["label"]

    if best_final_price > 0.5:
        return best_final

    # Last fallback: last_trade_price (unreliable for old markets)
    best = max(buckets, key=lambda b: b.get("last_price") or 0)
    if (best.get("last_price") or 0) > 0.5:
        return best["label"]
    return None


def _temp_in_bucket(temp_c: float, bucket: dict) -> bool:
    """Check if temp falls in bucket."""
    parsed = {
        "unit": bucket["unit"], "low": bucket["low"], "high": bucket["high"],
        "is_lower_bound": bucket["is_lower"], "is_upper_bound": bucket["is_upper"],
    }
    return _bucket_contains_temp_c(parsed, temp_c)


def analyze_real_prices(daily: pd.DataFrame, pm: pd.DataFrame) -> pd.DataFrame:
    """Main analysis: for each day, compute real prices and outcomes.

    For each day where we have both forecast data and PM data:
    - Find which PM bucket the ECMWF forecast maps to
    - Find which PM bucket the GFS forecast maps to
    - Get the real price of that bucket at J-1 evening (~18h before) and J-2 evening (~36h)
    - Determine if models converge on same PM bucket
    - Determine if the bet would have won
    - Compute real P/L

    Returns DataFrame with one row per day.
    """
    rows = []

    for _, day_row in daily.iterrows():
        d = day_row["date"]
        d_date = d.date() if hasattr(d, "date") else d

        pm_row = pm[pm["date"].dt.date == d_date]
        if pm_row.empty or not pm_row.iloc[0]["event_found"]:
            continue

        buckets_json = pm_row.iloc[0]["buckets_json"]
        if pd.isna(buckets_json):
            continue

        buckets = json.loads(buckets_json)
        if not buckets:
            continue

        ecmwf_fc = day_row.get("ecmwf_forecast")
        gfs_fc = day_row.get("gfs_forecast")
        actual = day_row.get("actual_temp")

        if pd.isna(ecmwf_fc) or pd.isna(gfs_fc) or pd.isna(actual):
            continue

        # Resolution time: end of day
        res_dt = datetime(d_date.year, d_date.month, d_date.day, 23, 59)

        # Target timestamps for price snapshots
        ts_18h = (res_dt - timedelta(hours=18)).timestamp()  # J-1 evening
        ts_36h = (res_dt - timedelta(hours=36)).timestamp()  # J-2 evening
        ts_6h = (res_dt - timedelta(hours=6)).timestamp()    # Day-of morning

        # Find bucket prices for each model forecast
        ecmwf_price_18h, ecmwf_bucket = _find_bucket_price(buckets, ecmwf_fc, ts_18h)
        gfs_price_18h, gfs_bucket = _find_bucket_price(buckets, gfs_fc, ts_18h)
        ecmwf_price_36h, _ = _find_bucket_price(buckets, ecmwf_fc, ts_36h)
        gfs_price_36h, _ = _find_bucket_price(buckets, gfs_fc, ts_36h)

        # Also get J-2 forecasts if available
        ecmwf_fc_j2 = day_row.get("ecmwf_forecast_j2")
        gfs_fc_j2 = day_row.get("gfs_forecast_j2")

        ecmwf_price_36h_j2, ecmwf_bucket_j2 = None, None
        gfs_price_36h_j2, gfs_bucket_j2 = None, None
        if not pd.isna(ecmwf_fc_j2) and not pd.isna(gfs_fc_j2):
            ecmwf_price_36h_j2, ecmwf_bucket_j2 = _find_bucket_price(buckets, ecmwf_fc_j2, ts_36h)
            gfs_price_36h_j2, gfs_bucket_j2 = _find_bucket_price(buckets, gfs_fc_j2, ts_36h)

        # Check convergence (same PM bucket for J-1 forecasts)
        converge_j1 = ecmwf_bucket is not None and gfs_bucket is not None and ecmwf_bucket == gfs_bucket

        # Check convergence for J-2 forecasts
        converge_j2 = (ecmwf_bucket_j2 is not None and gfs_bucket_j2 is not None
                       and ecmwf_bucket_j2 == gfs_bucket_j2)

        # Determine winning bucket from PM resolution
        winning_bucket = _find_winning_bucket(buckets)

        # Check if actual temp falls in the bet bucket
        # For convergence strategy, we bet on the consensus bucket
        consensus_bucket = ecmwf_bucket if converge_j1 else None

        win_convergence = (consensus_bucket is not None
                          and winning_bucket is not None
                          and consensus_bucket == winning_bucket)

        win_ecmwf = (ecmwf_bucket is not None
                     and winning_bucket is not None
                     and ecmwf_bucket == winning_bucket)

        win_gfs = (gfs_bucket is not None
                   and winning_bucket is not None
                   and gfs_bucket == winning_bucket)

        # Also check using actual temp directly (for comparison)
        actual_in_ecmwf_bucket = False
        actual_in_gfs_bucket = False
        if ecmwf_bucket is not None:
            for b in buckets:
                if b["label"] == ecmwf_bucket:
                    actual_in_ecmwf_bucket = _temp_in_bucket(actual, b)
                    break
        if gfs_bucket is not None:
            for b in buckets:
                if b["label"] == gfs_bucket:
                    actual_in_gfs_bucket = _temp_in_bucket(actual, b)
                    break

        unit = buckets[0]["unit"]

        row = {
            "date": d_date,
            "unit": unit,
            "num_buckets": len(buckets),
            "ecmwf_forecast": ecmwf_fc,
            "gfs_forecast": gfs_fc,
            "actual_temp": actual,
            "ecmwf_bucket": ecmwf_bucket,
            "gfs_bucket": gfs_bucket,
            "winning_bucket": winning_bucket,
            "converge_j1": converge_j1,
            "converge_j2": converge_j2,
            # Real prices at different timings
            "ecmwf_price_18h": ecmwf_price_18h,
            "ecmwf_price_36h": ecmwf_price_36h,
            "gfs_price_18h": gfs_price_18h,
            "gfs_price_36h": gfs_price_36h,
            # Outcomes
            "win_ecmwf": win_ecmwf,
            "win_gfs": win_gfs,
            "win_convergence": win_convergence,
            # Cross-check with actual temp
            "actual_in_ecmwf_bucket": actual_in_ecmwf_bucket,
            "actual_in_gfs_bucket": actual_in_gfs_bucket,
        }

        rows.append(row)

    return pd.DataFrame(rows)


def compute_real_pnl(df: pd.DataFrame) -> dict:
    """Compute P/L with real Polymarket prices for various strategies."""
    results = {}

    # --- Strategy: Naive ECMWF with real prices (J-1 evening, 18h) ---
    ecmwf_trades = df[df["ecmwf_price_18h"].notna()].copy()
    ecmwf_trades["pnl"] = ecmwf_trades.apply(
        lambda r: (1 - r["ecmwf_price_18h"]) * 0.98 if r["win_ecmwf"] else -r["ecmwf_price_18h"],
        axis=1,
    )
    results["naive_ecmwf_real_18h"] = {
        "trades": len(ecmwf_trades),
        "wins": ecmwf_trades["win_ecmwf"].sum(),
        "win_rate": ecmwf_trades["win_ecmwf"].mean() if len(ecmwf_trades) > 0 else 0,
        "total_pnl": ecmwf_trades["pnl"].sum(),
        "avg_price": ecmwf_trades["ecmwf_price_18h"].mean(),
        "avg_pnl_per_trade": ecmwf_trades["pnl"].mean() if len(ecmwf_trades) > 0 else 0,
        "monthly_pnl": ecmwf_trades.groupby(ecmwf_trades["date"].apply(lambda d: d.month))["pnl"].sum().to_dict(),
    }

    # --- Strategy: Convergence with real prices (J-1 evening, 18h) ---
    conv = df[df["converge_j1"] & df["ecmwf_price_18h"].notna()].copy()
    conv["pnl"] = conv.apply(
        lambda r: (1 - r["ecmwf_price_18h"]) * 0.98 if r["win_convergence"] else -r["ecmwf_price_18h"],
        axis=1,
    )
    results["convergence_real_18h"] = {
        "trades": len(conv),
        "wins": conv["win_convergence"].sum(),
        "win_rate": conv["win_convergence"].mean() if len(conv) > 0 else 0,
        "total_pnl": conv["pnl"].sum(),
        "avg_price": conv["ecmwf_price_18h"].mean(),
        "avg_pnl_per_trade": conv["pnl"].mean() if len(conv) > 0 else 0,
        "monthly_pnl": conv.groupby(conv["date"].apply(lambda d: d.month))["pnl"].sum().to_dict(),
    }

    # --- Strategy: Convergence with real prices (J-2 evening, 36h) ---
    conv36 = df[df["converge_j1"] & df["ecmwf_price_36h"].notna()].copy()
    conv36["pnl"] = conv36.apply(
        lambda r: (1 - r["ecmwf_price_36h"]) * 0.98 if r["win_convergence"] else -r["ecmwf_price_36h"],
        axis=1,
    )
    results["convergence_real_36h"] = {
        "trades": len(conv36),
        "wins": conv36["win_convergence"].sum(),
        "win_rate": conv36["win_convergence"].mean() if len(conv36) > 0 else 0,
        "total_pnl": conv36["pnl"].sum(),
        "avg_price": conv36["ecmwf_price_36h"].mean(),
        "avg_pnl_per_trade": conv36["pnl"].mean() if len(conv36) > 0 else 0,
        "monthly_pnl": conv36.groupby(conv36["date"].apply(lambda d: d.month))["pnl"].sum().to_dict(),
    }

    # --- Strategy: Convergence with price ceiling (only buy if price < threshold) ---
    for threshold in [0.40, 0.50, 0.60]:
        filtered = conv[conv["ecmwf_price_18h"] < threshold].copy()
        if len(filtered) > 0:
            filtered["pnl"] = filtered.apply(
                lambda r: (1 - r["ecmwf_price_18h"]) * 0.98 if r["win_convergence"] else -r["ecmwf_price_18h"],
                axis=1,
            )
            results[f"conv_ceiling_{int(threshold*100)}c"] = {
                "trades": len(filtered),
                "wins": filtered["win_convergence"].sum(),
                "win_rate": filtered["win_convergence"].mean(),
                "total_pnl": filtered["pnl"].sum(),
                "avg_price": filtered["ecmwf_price_18h"].mean(),
                "avg_pnl_per_trade": filtered["pnl"].mean(),
            }

    return results


def print_report(analysis: pd.DataFrame, pnl_results: dict) -> None:
    """Print the V2 analysis report to stdout."""
    print("\n" + "=" * 70)
    print("V2 BACKTEST — REAL POLYMARKET PRICES")
    print("=" * 70)

    # Data coverage
    total = len(analysis)
    f_days = (analysis["unit"] == "F").sum()
    c_days = (analysis["unit"] == "C").sum()
    print(f"\nData coverage: {total} days ({f_days} in °F, {c_days} in °C)")

    # Price distribution
    print(f"\n--- PRICE DISTRIBUTION (consensus bucket) ---")
    for timing, col in [("J-1 soir (18h)", "ecmwf_price_18h"), ("J-2 soir (36h)", "ecmwf_price_36h")]:
        valid = analysis[col].dropna()
        if len(valid) > 0:
            print(f"  {timing}: mean={valid.mean():.3f} | median={valid.median():.3f} | "
                  f"std={valid.std():.3f} | min={valid.min():.3f} | max={valid.max():.3f}")

    # Compare prices: convergence days vs all days
    conv_days = analysis[analysis["converge_j1"]]
    div_days = analysis[~analysis["converge_j1"]]
    if len(conv_days) > 0:
        conv_prices = conv_days["ecmwf_price_18h"].dropna()
        print(f"\n  Convergence days ({len(conv_days)}): avg price = {conv_prices.mean():.3f}")
    if len(div_days) > 0:
        div_prices = div_days["ecmwf_price_18h"].dropna()
        print(f"  Divergence days ({len(div_days)}): avg ECMWF price = {div_prices.mean():.3f}")

    # Win rates with real PM resolution
    print(f"\n--- WIN RATES (PM resolution) ---")
    ecmwf_valid = analysis["win_ecmwf"].notna()
    gfs_valid = analysis["win_gfs"].notna()
    print(f"  ECMWF bucket accuracy (PM buckets): {analysis.loc[ecmwf_valid, 'win_ecmwf'].mean()*100:.1f}%")
    print(f"  GFS bucket accuracy (PM buckets):   {analysis.loc[gfs_valid, 'win_gfs'].mean()*100:.1f}%")

    if len(conv_days) > 0:
        print(f"  Convergence win rate:               {conv_days['win_convergence'].mean()*100:.1f}%")

    # Cross-check: PM resolution vs our actual temp
    ecmwf_match = analysis["actual_in_ecmwf_bucket"].mean() * 100
    gfs_match = analysis["actual_in_gfs_bucket"].mean() * 100
    pm_match = analysis["win_ecmwf"].mean() * 100
    print(f"\n  (Cross-check: ECMWF vs our actual = {ecmwf_match:.1f}%, vs PM resolution = {pm_match:.1f}%)")
    print(f"  (Difference = {abs(ecmwf_match - pm_match):.1f}pp — this is the resolution source gap)")

    # P/L results
    print(f"\n--- P/L WITH REAL PRICES ($1/trade, 2% fees on wins) ---")
    for name, r in pnl_results.items():
        print(f"\n  {name}:")
        print(f"    Trades: {r['trades']} | Wins: {r['wins']} ({r['win_rate']*100:.1f}%)")
        print(f"    Avg buy price: {r['avg_price']:.3f}")
        print(f"    P/L per trade: ${r['avg_pnl_per_trade']:+.4f}")
        print(f"    Total P/L: ${r['total_pnl']:+.2f}")

    # Compare simulated vs real
    print(f"\n--- SIMULATED vs REAL COMPARISON ---")
    print(f"  {'Strategy':<30} {'Sim P/L':>10} {'Real P/L':>10} {'Delta':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")

    # Load simulated results for comparison
    sim_trades = pd.read_csv(DATA_PROCESSED_DIR / "trades_log.csv")
    sim_conv = sim_trades[sim_trades["strategy"] == "convergence"]
    sim_ecmwf = sim_trades[sim_trades["strategy"] == "naive_ecmwf"]

    if "naive_ecmwf_real_18h" in pnl_results:
        sim_pnl = sim_ecmwf["pnl"].sum()
        real_pnl = pnl_results["naive_ecmwf_real_18h"]["total_pnl"]
        print(f"  {'Naive ECMWF':<30} ${sim_pnl:>+9.2f} ${real_pnl:>+9.2f} ${real_pnl-sim_pnl:>+9.2f}")

    if "convergence_real_18h" in pnl_results:
        sim_pnl = sim_conv["pnl"].sum()
        real_pnl = pnl_results["convergence_real_18h"]["total_pnl"]
        print(f"  {'Convergence (18h)':<30} ${sim_pnl:>+9.2f} ${real_pnl:>+9.2f} ${real_pnl-sim_pnl:>+9.2f}")

    if "convergence_real_36h" in pnl_results:
        real_pnl = pnl_results["convergence_real_36h"]["total_pnl"]
        print(f"  {'Convergence (36h, early)':<30} {'N/A':>10} ${real_pnl:>+9.2f} {'':>10}")

    # Breakdown by unit (°F period vs °C period)
    print(f"\n--- BREAKDOWN: °F PERIOD vs °C PERIOD ---")
    for unit_label, unit_val in [("°F (Mar-Dec 2025)", "F"), ("°C (Dec 2025-Mar 2026)", "C")]:
        subset = analysis[analysis["unit"] == unit_val]
        if len(subset) == 0:
            continue
        conv_sub = subset[subset["converge_j1"]]
        if len(conv_sub) > 0:
            wr = conv_sub["win_convergence"].mean() * 100
            prices = conv_sub["ecmwf_price_18h"].dropna()
            avg_p = prices.mean() if len(prices) > 0 else 0
            # Compute P/L
            pnl_vals = conv_sub.apply(
                lambda r: (1 - r["ecmwf_price_18h"]) * 0.98 if r["win_convergence"] else -r["ecmwf_price_18h"]
                if not pd.isna(r["ecmwf_price_18h"]) else 0,
                axis=1,
            )
            print(f"  {unit_label}: {len(conv_sub)} conv trades, WR={wr:.1f}%, avg_price={avg_p:.3f}, P/L=${pnl_vals.sum():+.2f}")


def generate_v2_plots(analysis: pd.DataFrame, pnl_results: dict) -> None:
    """Generate V2 analysis plots."""
    try:
        plt.style.use(PLOT_STYLE)
    except OSError:
        pass

    OUTPUT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Plot 1: Real price distribution ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Histogram of consensus bucket prices at 18h
    conv = analysis[analysis["converge_j1"]]
    prices_18h = conv["ecmwf_price_18h"].dropna()
    prices_36h = conv["ecmwf_price_36h"].dropna()

    axes[0].hist(prices_18h, bins=30, color=PLOT_COLORS["ecmwf"], alpha=0.7, edgecolor="white")
    axes[0].axvline(0.30, color="gray", linestyle="--", label="Sim price (30¢)")
    axes[0].axvline(prices_18h.median(), color="black", linestyle="-", linewidth=2, label=f"Median ({prices_18h.median():.2f})")
    axes[0].set_title("Prix réel bucket consensus — J-1 soir (18h)")
    axes[0].set_xlabel("Prix ($)")
    axes[0].set_ylabel("Fréquence")
    axes[0].legend()

    axes[1].hist(prices_36h, bins=30, color=PLOT_COLORS["gfs"], alpha=0.7, edgecolor="white")
    axes[1].axvline(0.20, color="gray", linestyle="--", label="Sim price (20¢)")
    axes[1].axvline(prices_36h.median(), color="black", linestyle="-", linewidth=2, label=f"Median ({prices_36h.median():.2f})")
    axes[1].set_title("Prix réel bucket consensus — J-2 soir (36h)")
    axes[1].set_xlabel("Prix ($)")
    axes[1].legend()

    # Scatter: price vs outcome
    conv_with_prices = conv[conv["ecmwf_price_18h"].notna()].copy()
    wins = conv_with_prices[conv_with_prices["win_convergence"]]
    losses = conv_with_prices[~conv_with_prices["win_convergence"]]
    axes[2].scatter(wins.index, wins["ecmwf_price_18h"], c=PLOT_COLORS["actual"], alpha=0.5, s=15, label=f"Win ({len(wins)})")
    axes[2].scatter(losses.index, losses["ecmwf_price_18h"], c=PLOT_COLORS["ecmwf"], alpha=0.5, s=15, label=f"Loss ({len(losses)})")
    axes[2].set_title("Prix d'achat — Wins vs Losses")
    axes[2].set_xlabel("Trade #")
    axes[2].set_ylabel("Prix d'achat ($)")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOTS_DIR / "v2_01_real_price_distribution.png", dpi=PLOT_DPI)
    plt.close()

    # --- Plot 2: Cumulative P/L comparison (simulated vs real) ---
    fig, ax = plt.subplots(figsize=(14, 6))

    # Simulated convergence P/L
    sim_trades = pd.read_csv(DATA_PROCESSED_DIR / "trades_log.csv", parse_dates=["date"])
    sim_conv = sim_trades[sim_trades["strategy"] == "convergence"].sort_values("date")
    ax.plot(sim_conv["date"], sim_conv["pnl"].cumsum(), label="Convergence (prix simulé 30¢)",
            color="gray", linestyle="--", alpha=0.7)

    # Real P/L at 18h
    conv_real = analysis[analysis["converge_j1"] & analysis["ecmwf_price_18h"].notna()].copy()
    conv_real = conv_real.sort_values("date")
    conv_real["pnl_18h"] = conv_real.apply(
        lambda r: (1 - r["ecmwf_price_18h"]) * 0.98 if r["win_convergence"] else -r["ecmwf_price_18h"],
        axis=1,
    )
    conv_real["pnl_36h"] = conv_real.apply(
        lambda r: (1 - r["ecmwf_price_36h"]) * 0.98 if r["win_convergence"] and not pd.isna(r["ecmwf_price_36h"]) else (
            -r["ecmwf_price_36h"] if not pd.isna(r["ecmwf_price_36h"]) else 0),
        axis=1,
    )

    dates = pd.to_datetime(conv_real["date"])
    ax.plot(dates, conv_real["pnl_18h"].cumsum(),
            label=f"Convergence (prix réel J-1, P/L=${conv_real['pnl_18h'].sum():+.1f})",
            color=PLOT_COLORS["ecmwf"], linewidth=2)
    ax.plot(dates, conv_real["pnl_36h"].cumsum(),
            label=f"Convergence (prix réel J-2, P/L=${conv_real['pnl_36h'].sum():+.1f})",
            color=PLOT_COLORS["gfs"], linewidth=2)

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("P/L cumulé : prix simulés vs prix réels Polymarket")
    ax.set_xlabel("Date")
    ax.set_ylabel("P/L cumulé ($)")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOTS_DIR / "v2_02_cumulative_pnl_real.png", dpi=PLOT_DPI)
    plt.close()

    # --- Plot 3: Monthly P/L comparison ---
    fig, ax = plt.subplots(figsize=(14, 6))
    conv_real["month"] = pd.to_datetime(conv_real["date"]).dt.to_period("M")
    monthly = conv_real.groupby("month").agg(
        pnl_18h=("pnl_18h", "sum"),
        pnl_36h=("pnl_36h", "sum"),
        trades=("pnl_18h", "count"),
        win_rate=("win_convergence", "mean"),
    ).reset_index()
    monthly["month_str"] = monthly["month"].astype(str)

    x = range(len(monthly))
    width = 0.35
    ax.bar([i - width/2 for i in x], monthly["pnl_18h"], width, label="Real J-1 (18h)", color=PLOT_COLORS["ecmwf"])
    ax.bar([i + width/2 for i in x], monthly["pnl_36h"], width, label="Real J-2 (36h)", color=PLOT_COLORS["gfs"])
    ax.set_xticks(list(x))
    ax.set_xticklabels(monthly["month_str"], rotation=45)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("P/L mensuel — prix réels Polymarket (convergence)")
    ax.set_ylabel("P/L ($)")
    ax.legend()

    # Add win rate as text on top
    for i, row in monthly.iterrows():
        ax.text(i, max(row["pnl_18h"], row["pnl_36h"]) + 0.3,
                f"{row['win_rate']*100:.0f}%", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOTS_DIR / "v2_03_monthly_pnl_real.png", dpi=PLOT_DPI)
    plt.close()

    logger.info("V2 plots saved to %s", OUTPUT_PLOTS_DIR)


def main():
    daily, pm = load_data()
    logger.info("Loaded %d daily rows, %d PM rows", len(daily), len(pm))

    analysis = analyze_real_prices(daily, pm)
    logger.info("Analysis produced %d rows", len(analysis))

    # Save analysis
    analysis.to_csv(DATA_PROCESSED_DIR / "v2_real_prices_analysis.csv", index=False)

    pnl_results = compute_real_pnl(analysis)
    print_report(analysis, pnl_results)
    generate_v2_plots(analysis, pnl_results)


if __name__ == "__main__":
    main()
