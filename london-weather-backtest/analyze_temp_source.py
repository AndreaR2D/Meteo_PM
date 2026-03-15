"""Analyze the impact of temperature source on backtest results.

Compares Open-Meteo best_match (our proxy) vs Open-Meteo Archive API
(actual observations) to quantify the bias from using the wrong source.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import OUTPUT_PLOTS_DIR, PLOT_COLORS, PLOT_DPI, PLOT_STYLE
from src.backtest.buckets import assign_bucket, generate_daily_buckets


def main() -> None:
    daily = pd.read_csv("data/processed/forecasts_daily.csv", parse_dates=["date"])

    # Load both temperature sources
    obs = pd.read_csv("data/raw/actuals_archive.csv", parse_dates=["date"])
    obs = obs.rename(columns={"actual_temp": "obs_temp"})

    # Load best_match source (may need to fetch if not cached)
    from src.data.fetch_actuals import fetch_actuals
    bm = fetch_actuals("2025-03-14", "2026-03-14", source="best_match")
    bm = bm.rename(columns={"actual_temp": "bm_temp"})

    merged = daily.merge(obs, on="date", how="inner").merge(bm, on="date", how="inner")

    diff = merged["obs_temp"] - merged["bm_temp"]
    abs_diff = diff.abs()

    print("=" * 60)
    print("TEMPERATURE SOURCE ANALYSIS")
    print("Archive (observations) vs Best_match (proxy)")
    print("=" * 60)
    print(f"\nJours compares: {len(merged)}")
    print(f"Biais moyen: {diff.mean():+.2f}C (obs - best_match)")
    print(f"Ecart abs moyen: {abs_diff.mean():.2f}C")
    print(f"Ecart abs median: {abs_diff.median():.2f}C")
    print(f"Ecart max: {abs_diff.max():.2f}C")
    print()

    for t in [0.5, 1.0, 1.5, 2.0]:
        pct = (abs_diff <= t).mean() * 100
        print(f"  |diff| <= {t}C: {pct:.1f}%")

    # Count bucket changes
    bucket_changes = 0
    total = 0
    for _, row in merged.iterrows():
        bm = row["bm_temp"]
        ob = row["obs_temp"]
        if pd.isna(bm) or pd.isna(ob):
            continue
        buckets = generate_daily_buckets(row["date"].month)
        bm_b = assign_bucket(bm, buckets)
        ob_b = assign_bucket(ob, buckets)
        total += 1
        if bm_b and ob_b and bm_b["index"] != ob_b["index"]:
            bucket_changes += 1

    print(f"\nBucket changes entre les 2 sources: {bucket_changes}/{total} ({bucket_changes/total*100:.1f}%)")

    # Win rate comparison
    print("\n--- CONVERGENCE WIN RATE ---")
    for source_name, temp_col in [
        ("best_match (proxy)", "bm_temp"),
        ("archive (observations)", "obs_temp"),
    ]:
        conv_total = 0
        conv_wins = 0
        for _, row in merged.iterrows():
            gfs = row.get("gfs_forecast")
            ecmwf = row.get("ecmwf_forecast")
            actual = row.get(temp_col)
            if pd.isna(gfs) or pd.isna(ecmwf) or pd.isna(actual):
                continue
            buckets = generate_daily_buckets(row["date"].month)
            gb = assign_bucket(gfs, buckets)
            eb = assign_bucket(ecmwf, buckets)
            ab = assign_bucket(actual, buckets)
            if gb and eb and ab and gb["index"] == eb["index"]:
                conv_total += 1
                if gb["index"] == ab["index"]:
                    conv_wins += 1
        wr = conv_wins / conv_total * 100 if conv_total > 0 else 0
        print(f"  {source_name}: {conv_wins}/{conv_total} = {wr:.1f}%")

    # On the days where bucket differs, who wins?
    print("\n--- JOURS OU LE BUCKET DIFFERE (convergence days only) ---")
    changed_total = 0
    win_bm_only = 0
    win_obs_only = 0
    win_both = 0
    win_neither = 0

    for _, row in merged.iterrows():
        gfs = row.get("gfs_forecast")
        ecmwf = row.get("ecmwf_forecast")
        bm_val = row.get("bm_temp")
        obs_val = row.get("obs_temp")
        if pd.isna(gfs) or pd.isna(ecmwf) or pd.isna(bm_val) or pd.isna(obs_val):
            continue
        buckets = generate_daily_buckets(row["date"].month)
        gb = assign_bucket(gfs, buckets)
        eb = assign_bucket(ecmwf, buckets)
        bm_b = assign_bucket(bm_val, buckets)
        ob_b = assign_bucket(obs_val, buckets)
        if not (gb and eb and bm_b and ob_b and gb["index"] == eb["index"]):
            continue
        if bm_b["index"] == ob_b["index"]:
            continue  # Same bucket, no issue

        changed_total += 1
        w_bm = gb["index"] == bm_b["index"]
        w_ob = gb["index"] == ob_b["index"]
        if w_bm and w_ob:
            win_both += 1
        elif w_bm and not w_ob:
            win_bm_only += 1
        elif w_ob and not w_bm:
            win_obs_only += 1
        else:
            win_neither += 1

    print(f"  Total jours with bucket mismatch: {changed_total}")
    print(f"  Win avec best_match seulement: {win_bm_only} (on perd ces trades avec obs)")
    print(f"  Win avec observations seulement: {win_obs_only} (on gagne ces trades avec obs)")
    print(f"  Win avec les deux: {win_both}")
    print(f"  Lose avec les deux: {win_neither}")
    print(f"  Net impact: {win_obs_only - win_bm_only:+d} trades")

    # MAE comparison
    print("\n--- MAE PAR MODELE ---")
    for model in ["gfs", "ecmwf"]:
        col = f"{model}_forecast"
        err_bm = (merged[col] - merged["bm_temp"]).abs().mean()
        err_obs = (merged[col] - merged["obs_temp"]).abs().mean()
        print(f"  {model.upper()}: vs best_match={err_bm:.2f}C | vs archive={err_obs:.2f}C")

    # === PLOT ===
    try:
        plt.style.use(PLOT_STYLE)
    except OSError:
        plt.style.use("seaborn-v0_8")

    OUTPUT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Scatter best_match vs archive
    ax = axes[0, 0]
    ax.scatter(merged["bm_temp"], merged["obs_temp"], alpha=0.4, s=12, color="#2196F3")
    lims = [merged[["bm_temp", "obs_temp"]].min().min() - 2,
            merged[["bm_temp", "obs_temp"]].max().max() + 2]
    ax.plot(lims, lims, "k--", alpha=0.5)
    ax.set_xlabel("Best_match (proxy) °C")
    ax.set_ylabel("Archive (observations) °C")
    ax.set_title("Best_match vs Observations")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # Panel 2: Histogram of differences
    ax = axes[0, 1]
    ax.hist(diff.dropna(), bins=40, color="#FF9800", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="black", linestyle="--", alpha=0.5)
    ax.axvline(diff.mean(), color="red", linestyle="-", label=f"Biais: {diff.mean():.2f}C")
    ax.set_xlabel("Ecart (obs - best_match) °C")
    ax.set_ylabel("Frequence")
    ax.set_title("Distribution des ecarts entre sources")
    ax.legend()

    # Panel 3: Absolute diff over time
    ax = axes[1, 0]
    ax.plot(merged["date"], abs_diff, alpha=0.3, color="#F44336")
    ma = abs_diff.rolling(14, min_periods=1).mean()
    ax.plot(merged["date"], ma, color="#F44336", linewidth=2, label="MA 14j")
    ax.set_xlabel("Date")
    ax.set_ylabel("|Ecart| °C")
    ax.set_title("Ecart absolu au fil du temps")
    ax.legend()

    # Panel 4: Monthly bucket change rate
    ax = axes[1, 1]
    merged_c = merged.copy()
    merged_c["month"] = merged_c["date"].dt.month
    monthly_changes = []
    for month in range(1, 13):
        m_data = merged_c[merged_c["month"] == month]
        if m_data.empty:
            continue
        changes = 0
        total_m = 0
        for _, row in m_data.iterrows():
            bm_val = row["bm_temp"]
            obs_val = row["obs_temp"]
            if pd.isna(bm_val) or pd.isna(obs_val):
                continue
            buckets = generate_daily_buckets(month)
            bm_b = assign_bucket(bm_val, buckets)
            ob_b = assign_bucket(obs_val, buckets)
            total_m += 1
            if bm_b and ob_b and bm_b["index"] != ob_b["index"]:
                changes += 1
        monthly_changes.append({
            "month": month,
            "change_pct": changes / total_m * 100 if total_m > 0 else 0,
        })

    mc_df = pd.DataFrame(monthly_changes)
    month_names = ["Jan", "Fev", "Mar", "Avr", "Mai", "Jun",
                   "Jul", "Aou", "Sep", "Oct", "Nov", "Dec"]
    ax.bar(
        [month_names[m - 1] for m in mc_df["month"]],
        mc_df["change_pct"],
        color="#9C27B0", alpha=0.7,
    )
    ax.set_ylabel("% jours avec bucket different")
    ax.set_title("Taux de changement de bucket par mois")
    ax.axhline(24, color="gray", linestyle="--", alpha=0.5, label="Moyenne 24%")
    ax.legend()

    fig.suptitle(
        "Analyse source de temperature: Best_match vs Observations reelles",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_PLOTS_DIR / "14_temp_source_analysis.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {OUTPUT_PLOTS_DIR / '14_temp_source_analysis.png'}")


if __name__ == "__main__":
    main()
