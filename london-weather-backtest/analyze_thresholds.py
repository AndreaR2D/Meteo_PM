"""Analyze the price threshold strategy across multiple scenarios.

Simulates realistic market prices and tests different entry thresholds
to find the sweet spot between trade frequency and alpha per trade.

Key assumptions:
- Market price for the consensus bucket follows a distribution
  based on how "obvious" the forecast is
- Earlier entry = lower price (market hasn't absorbed forecast yet)
- Polymarket charges 2% fee on winnings
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import OUTPUT_PLOTS_DIR, PLOT_DPI, PLOT_STYLE
from src.backtest.buckets import assign_bucket, generate_daily_buckets

PM_FEE = 0.02  # 2% fee on winnings


def simulate_market_prices(daily_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Simulate realistic Polymarket prices for the consensus bucket.

    The price depends on:
    - How close GFS and ECMWF forecasts are (tighter = higher price)
    - Time of day / lead time (we simulate J-1 evening prices)
    - Random noise (market microstructure, liquidity)

    We model the consensus bucket price as:
        base_price ~ Normal(0.45, 0.12) clipped to [0.15, 0.80]

    This reflects real PM weather markets where consensus outcomes
    typically trade between 30-65 cents the evening before.

    Args:
        daily_df: Daily DataFrame with forecasts and actuals.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with added 'market_price' and 'converge' columns.
    """
    rng = np.random.default_rng(seed)
    df = daily_df.copy()

    prices = []
    converge_flags = []

    for _, row in df.iterrows():
        gfs = row.get("gfs_forecast")
        ecmwf = row.get("ecmwf_forecast")

        if pd.isna(gfs) or pd.isna(ecmwf):
            prices.append(np.nan)
            converge_flags.append(False)
            continue

        buckets = generate_daily_buckets(row["date"].month)
        gb = assign_bucket(gfs, buckets)
        eb = assign_bucket(ecmwf, buckets)

        if gb is None or eb is None:
            prices.append(np.nan)
            converge_flags.append(False)
            continue

        converge = gb["index"] == eb["index"]
        converge_flags.append(converge)

        if converge:
            # Semi-efficient market model:
            # The market "knows" convergence = higher probability.
            # Base price reflects partial market efficiency (~65-70%
            # of the true probability), with noise for timing/liquidity.
            # Tighter forecast spread => market more confident => higher price.
            forecast_spread = abs(gfs - ecmwf)
            # spread 0°C => base ~0.65, spread 1.5°C => base ~0.53
            base = 0.65 - forecast_spread * 0.08
            # Noise from market microstructure, time of entry, liquidity
            noise = rng.normal(0, 0.10)
            price = np.clip(base + noise, 0.25, 0.85)
        else:
            price = np.nan  # We don't trade when models diverge

        prices.append(price)

    df["market_price"] = prices
    df["converge"] = converge_flags

    return df


def run_threshold_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Run the threshold strategy at multiple price ceilings.

    Args:
        df: DataFrame with market_price, converge, forecasts, and actuals.

    Returns:
        Summary DataFrame with metrics per threshold.
    """
    thresholds = np.arange(0.25, 0.76, 0.05)
    results = []

    conv_df = df[df["converge"] & df["market_price"].notna()].copy()

    for threshold in thresholds:
        eligible = conv_df[conv_df["market_price"] <= threshold]

        if eligible.empty:
            results.append({
                "threshold": threshold,
                "trades": 0,
                "wins": 0,
                "win_rate": 0,
                "avg_price": 0,
                "pnl_total": 0,
                "pnl_per_trade": 0,
                "skip_pct": 100,
            })
            continue

        wins = 0
        total_pnl = 0
        prices_paid = []

        for _, row in eligible.iterrows():
            buckets = generate_daily_buckets(row["date"].month)
            gfs_bucket = assign_bucket(row["gfs_forecast"], buckets)
            actual_bucket = assign_bucket(row["actual_temp"], buckets)

            if gfs_bucket is None or actual_bucket is None:
                continue

            price = row["market_price"]
            prices_paid.append(price)
            win = gfs_bucket["index"] == actual_bucket["index"]

            if win:
                wins += 1
                gross = 1.0 - price
                total_pnl += gross * (1 - PM_FEE)  # Net of fees
            else:
                total_pnl -= price

        n_trades = len(prices_paid)
        total_conv_days = len(conv_df)

        results.append({
            "threshold": threshold,
            "trades": n_trades,
            "wins": wins,
            "win_rate": wins / n_trades if n_trades > 0 else 0,
            "avg_price": np.mean(prices_paid) if prices_paid else 0,
            "pnl_total": total_pnl,
            "pnl_per_trade": total_pnl / n_trades if n_trades > 0 else 0,
            "skip_pct": (1 - n_trades / total_conv_days) * 100 if total_conv_days > 0 else 100,
        })

    return pd.DataFrame(results)


def run_montecarlo_thresholds(
    daily_df: pd.DataFrame, n_sims: int = 200
) -> dict[float, list[float]]:
    """Run Monte Carlo simulations with different random market prices.

    Args:
        daily_df: Daily DataFrame with forecasts and actuals.
        n_sims: Number of simulations to run.

    Returns:
        Dict mapping threshold to list of total P/L across simulations.
    """
    thresholds = np.arange(0.25, 0.76, 0.05)
    mc_results: dict[float, list[float]] = {t: [] for t in thresholds}

    for sim in range(n_sims):
        df = simulate_market_prices(daily_df, seed=sim)
        summary = run_threshold_analysis(df)

        for _, row in summary.iterrows():
            mc_results[row["threshold"]].append(row["pnl_total"])

    return mc_results


def plot_threshold_analysis(summary: pd.DataFrame) -> None:
    """Plot the threshold analysis results."""
    try:
        plt.style.use(PLOT_STYLE)
    except OSError:
        plt.style.use("seaborn-v0_8")

    OUTPUT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    valid = summary[summary["trades"] > 0]

    # Panel 1: Trades count vs threshold
    ax = axes[0, 0]
    ax.bar(valid["threshold"], valid["trades"], width=0.04, color="#2196F3", alpha=0.8)
    ax.set_xlabel("Seuil de prix max")
    ax.set_ylabel("Nombre de trades")
    ax.set_title("Nombre de trades selon le seuil d'entrée")
    for _, row in valid.iterrows():
        ax.text(row["threshold"], row["trades"] + 2, f'{row["trades"]:.0f}',
                ha="center", fontsize=8)

    # Panel 2: Win rate vs threshold
    ax = axes[0, 1]
    ax.plot(valid["threshold"], valid["win_rate"] * 100, "o-", color="#4CAF50", linewidth=2)
    ax.axhline(74.2, color="gray", linestyle="--", alpha=0.5, label="WR global convergence")
    ax.set_xlabel("Seuil de prix max")
    ax.set_ylabel("Win Rate (%)")
    ax.set_title("Win Rate selon le seuil (les prix bas = jours plus incertains ?)")
    ax.legend()

    # Panel 3: P/L per trade vs threshold (THE KEY CHART)
    ax = axes[1, 0]
    colors = ["#4CAF50" if v >= 0 else "#F44336" for v in valid["pnl_per_trade"]]
    ax.bar(valid["threshold"], valid["pnl_per_trade"], width=0.04, color=colors, alpha=0.8)
    ax.axhline(0, color="black", linestyle="-", alpha=0.3)
    ax.set_xlabel("Seuil de prix max")
    ax.set_ylabel("P/L moyen par trade ($)")
    ax.set_title("Alpha par trade selon le seuil (net de fees 2%)")
    for _, row in valid.iterrows():
        offset = 0.005 if row["pnl_per_trade"] >= 0 else -0.015
        ax.text(row["threshold"], row["pnl_per_trade"] + offset,
                f'${row["pnl_per_trade"]:.3f}', ha="center", fontsize=8)

    # Panel 4: Total P/L vs threshold
    ax = axes[1, 1]
    colors = ["#4CAF50" if v >= 0 else "#F44336" for v in valid["pnl_total"]]
    ax.bar(valid["threshold"], valid["pnl_total"], width=0.04, color=colors, alpha=0.8)
    ax.axhline(0, color="black", linestyle="-", alpha=0.3)
    ax.set_xlabel("Seuil de prix max")
    ax.set_ylabel("P/L total ($)")
    ax.set_title("P/L total annuel selon le seuil (net de fees)")
    for _, row in valid.iterrows():
        offset = 1 if row["pnl_total"] >= 0 else -4
        ax.text(row["threshold"], row["pnl_total"] + offset,
                f'${row["pnl_total"]:.0f}', ha="center", fontsize=8)

    fig.suptitle(
        "Analyse seuil de prix : convergence + price ceiling\n"
        "(prix simulés ~ N(0.50, 0.10), fees PM 2%, $1/trade)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_PLOTS_DIR / "12_threshold_analysis.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PLOTS_DIR / '12_threshold_analysis.png'}")


def plot_montecarlo(mc_results: dict[float, list[float]]) -> None:
    """Plot Monte Carlo distribution of P/L by threshold."""
    try:
        plt.style.use(PLOT_STYLE)
    except OSError:
        plt.style.use("seaborn-v0_8")

    OUTPUT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    thresholds = sorted(mc_results.keys())
    # Filter to interesting range
    thresholds = [t for t in thresholds if 0.35 <= t <= 0.70]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: Box plot of P/L distribution by threshold
    ax = axes[0]
    data = [mc_results[t] for t in thresholds]
    bp = ax.boxplot(data, labels=[f"{t:.0%}" for t in thresholds], patch_artist=True)
    for i, patch in enumerate(bp["boxes"]):
        median = np.median(mc_results[thresholds[i]])
        color = "#4CAF50" if median > 0 else "#F44336"
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.axhline(0, color="black", linestyle="-", alpha=0.3)
    ax.set_xlabel("Seuil de prix max")
    ax.set_ylabel("P/L total ($)")
    ax.set_title("Distribution Monte Carlo du P/L (200 sims)")

    # Panel 2: Probability of profit by threshold
    ax = axes[1]
    prob_profit = [np.mean([p > 0 for p in mc_results[t]]) * 100 for t in thresholds]
    median_pnl = [np.median(mc_results[t]) for t in thresholds]
    ax.bar([f"{t:.0%}" for t in thresholds], prob_profit,
           color=["#4CAF50" if p > 50 else "#F44336" for p in prob_profit], alpha=0.8)
    ax.axhline(50, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Seuil de prix max")
    ax.set_ylabel("Probabilite de profit (%)")
    ax.set_title("Probabilite d'etre rentable sur 1 an")
    for i, (p, m) in enumerate(zip(prob_profit, median_pnl)):
        ax.text(i, p + 1, f"{p:.0f}%\n(med ${m:.0f})", ha="center", fontsize=8)

    fig.suptitle(
        "Monte Carlo : robustesse de la strategie selon le seuil de prix\n"
        "(200 simulations avec prix aleatoires ~ N(0.50, 0.10))",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_PLOTS_DIR / "13_montecarlo_thresholds.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PLOTS_DIR / '13_montecarlo_thresholds.png'}")


def simulate_market_prices_parametric(
    daily_df: pd.DataFrame,
    base_price: float = 0.65,
    noise_std: float = 0.10,
    slippage: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate market prices with configurable efficiency level.

    Args:
        daily_df: Daily DataFrame with forecasts and actuals.
        base_price: Base market price for consensus bucket (higher = more efficient).
        noise_std: Standard deviation of price noise.
        slippage: Fixed slippage cost added to buy price.
        seed: Random seed.

    Returns:
        DataFrame with market_price and converge columns.
    """
    rng = np.random.default_rng(seed)
    df = daily_df.copy()

    prices = []
    converge_flags = []

    for _, row in df.iterrows():
        gfs = row.get("gfs_forecast")
        ecmwf = row.get("ecmwf_forecast")
        if pd.isna(gfs) or pd.isna(ecmwf):
            prices.append(np.nan)
            converge_flags.append(False)
            continue
        buckets = generate_daily_buckets(row["date"].month)
        gb = assign_bucket(gfs, buckets)
        eb = assign_bucket(ecmwf, buckets)
        if gb is None or eb is None:
            prices.append(np.nan)
            converge_flags.append(False)
            continue
        converge = gb["index"] == eb["index"]
        converge_flags.append(converge)
        if converge:
            spread = abs(gfs - ecmwf)
            price = base_price - spread * 0.08 + rng.normal(0, noise_std) + slippage
            prices.append(np.clip(price, 0.20, 0.90))
        else:
            prices.append(np.nan)

    df["market_price"] = prices
    df["converge"] = converge_flags
    return df


def main() -> None:
    """Run the full threshold and Monte Carlo analysis."""
    daily = pd.read_csv("data/processed/forecasts_daily.csv", parse_dates=["date"])

    print("=" * 60)
    print("PRICE THRESHOLD ANALYSIS")
    print("=" * 60)

    # Single run with default prices
    print("\n--- Single simulation (seed=42) ---")
    df = simulate_market_prices(daily)

    conv_days = df[df["converge"]]
    print(f"\nSimulated market prices for convergence days:")
    print(f"  Count: {conv_days['market_price'].notna().sum()}")
    print(f"  Mean:  {conv_days['market_price'].mean():.3f}")
    print(f"  Std:   {conv_days['market_price'].std():.3f}")
    print(f"  Min:   {conv_days['market_price'].min():.3f}")
    print(f"  Max:   {conv_days['market_price'].max():.3f}")

    summary = run_threshold_analysis(df)

    print(f"\n{'Seuil':>8} | {'Trades':>6} | {'WR':>6} | {'Prix moy':>9} | {'P/L/trade':>10} | {'P/L total':>10} | {'Skip':>6}")
    print("-" * 75)
    for _, row in summary.iterrows():
        if row["trades"] > 0:
            print(f"  {row['threshold']:.0%}   | {row['trades']:>6.0f} | {row['win_rate']*100:>5.1f}% | "
                  f"  {row['avg_price']:>6.3f} | {row['pnl_per_trade']:>+9.3f} | {row['pnl_total']:>+9.1f} | {row['skip_pct']:>5.0f}%")

    plot_threshold_analysis(summary)

    # Monte Carlo
    print("\n--- Monte Carlo (200 simulations) ---")
    mc_results = run_montecarlo_thresholds(daily, n_sims=200)

    print(f"\n{'Seuil':>8} | {'P/L median':>10} | {'P/L P5':>8} | {'P/L P95':>8} | {'Prob profit':>11}")
    print("-" * 60)
    for t in sorted(mc_results.keys()):
        if 0.35 <= t <= 0.70:
            vals = mc_results[t]
            print(f"  {t:.0%}   | {np.median(vals):>+9.1f} | {np.percentile(vals,5):>+7.1f} | "
                  f"{np.percentile(vals,95):>+7.1f} | {np.mean([v>0 for v in vals])*100:>9.0f}%")

    plot_montecarlo(mc_results)

    # Find sweet spot
    print("\n--- SWEET SPOT ---")
    best_t = None
    best_score = -999
    for t in sorted(mc_results.keys()):
        if 0.35 <= t <= 0.70:
            vals = mc_results[t]
            median = np.median(vals)
            prob_profit = np.mean([v > 0 for v in vals])
            # Score: balance between median P/L and consistency
            score = median * prob_profit
            if score > best_score:
                best_score = score
                best_t = t

    if best_t is not None:
        vals = mc_results[best_t]
        print(f"  Seuil optimal: {best_t:.0%}")
        print(f"  P/L median:    ${np.median(vals):+.1f}")
        print(f"  P/L P5-P95:    ${np.percentile(vals,5):+.1f} a ${np.percentile(vals,95):+.1f}")
        print(f"  Prob profit:   {np.mean([v>0 for v in vals])*100:.0f}%")

    # =========================================================
    # STRESS TEST: Multiple market efficiency scenarios
    # =========================================================
    print("\n" + "=" * 60)
    print("STRESS TEST: Scenarios de marche")
    print("=" * 60)

    scenarios = [
        ("Marche lent (ton edge max)",       0.55, 0.12, 0.00),
        ("Marche semi-efficient",            0.65, 0.10, 0.00),
        ("Marche efficient",                 0.72, 0.08, 0.00),
        ("Marche efficient + slippage 3c",   0.72, 0.08, 0.03),
        ("Marche tres efficient",            0.76, 0.06, 0.00),
        ("Marche tres efficient + slip 3c",  0.76, 0.06, 0.03),
    ]

    threshold = 0.60  # Fixed threshold for comparison

    print(f"\nSeuil fixe: {threshold:.0%} | Fees PM: 2% | 200 sims Monte Carlo")
    print(f"\n{'Scenario':<35} | {'Trades':>6} | {'WR':>5} | {'Prix moy':>8} | {'P/L med':>8} | {'P5':>6} | {'P95':>6} | {'Prob+':>5}")
    print("-" * 100)

    for name, base, noise, slip in scenarios:
        pnls = []
        for sim_seed in range(200):
            sim_df = simulate_market_prices_parametric(
                daily, base_price=base, noise_std=noise, slippage=slip, seed=sim_seed
            )
            sim_summary = run_threshold_analysis(sim_df)
            row = sim_summary[sim_summary["threshold"].between(threshold - 0.01, threshold + 0.01)]
            if not row.empty:
                pnls.append(row.iloc[0]["pnl_total"])
            else:
                pnls.append(0.0)

        # Get typical stats from one run
        ref_df = simulate_market_prices_parametric(daily, base, noise, slip, seed=42)
        ref_summary = run_threshold_analysis(ref_df)
        ref_row = ref_summary[ref_summary["threshold"].between(threshold - 0.01, threshold + 0.01)]
        trades = ref_row.iloc[0]["trades"] if not ref_row.empty else 0
        wr = ref_row.iloc[0]["win_rate"] * 100 if not ref_row.empty else 0
        avg_p = ref_row.iloc[0]["avg_price"] if not ref_row.empty else 0

        prob_profit = np.mean([p > 0 for p in pnls]) * 100
        print(f"  {name:<33} | {trades:>6.0f} | {wr:>4.0f}% | {avg_p:>7.3f} | "
              f"${np.median(pnls):>+6.1f} | ${np.percentile(pnls,5):>+5.1f} | "
              f"${np.percentile(pnls,95):>+5.1f} | {prob_profit:>4.0f}%")

    print("\n--- CONCLUSION ---")
    print("Le win rate de 74% sur la convergence est un fait empirique solide.")
    print("La vraie question est: a quel prix le marche price-t-il le consensus?")
    print("Si le consensus est price a 70%+ le soir J-1, l'edge est quasi nul.")
    print("L'alpha reel vient de: parier AVANT que le marche n'absorbe les forecasts.")


if __name__ == "__main__":
    main()
