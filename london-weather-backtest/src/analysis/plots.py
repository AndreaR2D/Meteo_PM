"""Matplotlib visualizations for the backtest results."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import OUTPUT_PLOTS_DIR, PLOT_COLORS, PLOT_DPI, PLOT_STYLE

logger = logging.getLogger(__name__)


def _setup_style() -> None:
    """Apply the project plot style."""
    try:
        plt.style.use(PLOT_STYLE)
    except OSError:
        plt.style.use("seaborn-v0_8")


def _save_fig(fig: plt.Figure, name: str) -> None:
    """Save figure to output directory."""
    OUTPUT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_PLOTS_DIR / f"{name}.png"
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot: %s", path)


def plot_forecast_vs_actual(daily_df: pd.DataFrame) -> None:
    """Plot 1: Scatter plot of forecast vs actual for each model."""
    _setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, model in zip(axes, ["gfs", "ecmwf"]):
        col = f"{model}_forecast"
        if col not in daily_df.columns:
            continue
        valid = daily_df.dropna(subset=[col, "actual_temp"])
        if valid.empty:
            ax.set_title(f"{model.upper()} — Pas de données")
            continue
        ax.scatter(
            valid["actual_temp"], valid[col],
            alpha=0.4, s=15, color=PLOT_COLORS[model], label=model.upper(),
        )
        all_vals = pd.concat([valid["actual_temp"], valid[col]]).dropna()
        lims = [all_vals.min() - 2, all_vals.max() + 2]
        ax.plot(lims, lims, "k--", alpha=0.5, label="Prédiction parfaite")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Température réelle (°C)")
        ax.set_ylabel("Prévision (°C)")
        ax.set_title(f"{model.upper()} — Prévision vs Réalité")
        ax.legend()

    fig.suptitle("Prévision vs Température réelle — Londres EGLC", fontsize=14)
    _save_fig(fig, "01_forecast_vs_actual")


def plot_error_timeseries(daily_df: pd.DataFrame) -> None:
    """Plot 2: Daily error time series with 7-day moving average."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(14, 5))

    for model in ["gfs", "ecmwf"]:
        error_col = f"{model}_error"
        if error_col not in daily_df.columns:
            continue
        ax.plot(
            daily_df["date"], daily_df[error_col],
            alpha=0.2, color=PLOT_COLORS[model],
        )
        ma = daily_df[error_col].rolling(7, min_periods=1).mean()
        ax.plot(
            daily_df["date"], ma,
            color=PLOT_COLORS[model], linewidth=2, label=f"{model.upper()} (MA 7j)",
        )

    ax.axhline(0, color="black", linestyle="-", alpha=0.3)
    ax.set_xlabel("Date")
    ax.set_ylabel("Erreur (°C)")
    ax.set_title("Erreur quotidienne des modèles — Londres EGLC")
    ax.legend()
    _save_fig(fig, "02_error_timeseries")


def plot_monthly_mae(model_metrics: dict) -> None:
    """Plot 3: Monthly MAE bar chart, GFS vs ECMWF side by side."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(12, 5))

    months = sorted(set().union(*(m.get("monthly_mae", {}).keys() for m in model_metrics.values())))
    x = np.arange(len(months))
    width = 0.35

    for i, model in enumerate(["gfs", "ecmwf"]):
        if model not in model_metrics:
            continue
        mae_vals = [model_metrics[model]["monthly_mae"].get(m, 0) for m in months]
        ax.bar(
            x + i * width, mae_vals, width,
            label=model.upper(), color=PLOT_COLORS[model],
        )

    month_labels = ["Jan", "Fév", "Mar", "Avr", "Mai", "Jun",
                     "Jul", "Aoû", "Sep", "Oct", "Nov", "Déc"]
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([month_labels[m - 1] for m in months])
    ax.set_ylabel("MAE (°C)")
    ax.set_title("Erreur absolue moyenne par mois — GFS vs ECMWF")
    ax.legend()
    _save_fig(fig, "03_monthly_mae")


def plot_bucket_accuracy_by_month(model_metrics: dict) -> None:
    """Plot 4: Monthly bucket accuracy for each model."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(12, 5))

    months = sorted(set().union(
        *(m.get("monthly_bucket_accuracy", {}).keys() for m in model_metrics.values())
    ))
    x = np.arange(len(months))
    width = 0.35

    for i, model in enumerate(["gfs", "ecmwf"]):
        if model not in model_metrics:
            continue
        acc_vals = [
            model_metrics[model]["monthly_bucket_accuracy"].get(m, 0) * 100
            for m in months
        ]
        ax.bar(
            x + i * width, acc_vals, width,
            label=model.upper(), color=PLOT_COLORS[model],
        )

    month_labels = ["Jan", "Fév", "Mar", "Avr", "Mai", "Jun",
                     "Jul", "Aoû", "Sep", "Oct", "Nov", "Déc"]
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([month_labels[m - 1] for m in months])
    ax.set_ylabel("Bucket Accuracy (%)")
    ax.set_title("Précision par tranche (bucket) par mois — GFS vs ECMWF")
    ax.legend()
    _save_fig(fig, "04_bucket_accuracy_monthly")


def plot_cumulative_pnl(trades_df: pd.DataFrame) -> None:
    """Plot 5: Cumulative P/L curve for each strategy."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(14, 6))

    trades = trades_df.copy()
    trades["strategy_base"] = trades["strategy"].apply(
        lambda s: "best_model" if s.startswith("best_model") else s
    )

    colors = {
        "naive_gfs": PLOT_COLORS["gfs"],
        "naive_ecmwf": PLOT_COLORS["ecmwf"],
        "best_model": "#FF9800",
        "convergence": "#9C27B0",
    }

    for strategy in ["naive_gfs", "naive_ecmwf", "best_model", "convergence"]:
        strat_trades = trades[trades["strategy_base"] == strategy].sort_values("date")
        if strat_trades.empty:
            continue
        cumulative = strat_trades["pnl"].cumsum()
        ax.plot(
            strat_trades["date"].values, cumulative.values,
            label=strategy.replace("_", " ").title(),
            color=colors.get(strategy, None),
            linewidth=2,
        )

    ax.axhline(0, color="black", linestyle="-", alpha=0.3)
    ax.set_xlabel("Date")
    ax.set_ylabel("P/L cumulé ($)")
    ax.set_title("P/L cumulé par stratégie — Backtest Londres")
    ax.legend()
    _save_fig(fig, "05_cumulative_pnl")


def plot_convergence_analysis(convergence_stats: dict) -> None:
    """Plot 6: Win rate when models converge vs diverge."""
    _setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart: win rate comparison
    labels = ["Convergence", "Divergence"]
    win_rates = [
        convergence_stats["convergence_win_rate"] * 100,
        convergence_stats["divergence_win_rate"] * 100,
    ]
    colors = ["#4CAF50", "#F44336"]
    axes[0].bar(labels, win_rates, color=colors)
    axes[0].set_ylabel("Win Rate (%)")
    axes[0].set_title("Win Rate: Convergence vs Divergence")
    for i, v in enumerate(win_rates):
        axes[0].text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")

    # Pie chart: proportion of convergence vs divergence days
    sizes = [
        convergence_stats["convergence_days"],
        convergence_stats["divergence_days"],
    ]
    if sum(sizes) > 0:
        axes[1].pie(
            sizes, labels=labels, colors=colors, autopct="%1.1f%%",
            startangle=90,
        )
    axes[1].set_title("Répartition des jours")

    fig.suptitle("Analyse de convergence GFS / ECMWF", fontsize=14)
    _save_fig(fig, "06_convergence_analysis")


def plot_error_distribution(daily_df: pd.DataFrame) -> None:
    """Plot 7: Histogram of errors for each model."""
    _setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, model in zip(axes, ["gfs", "ecmwf"]):
        error_col = f"{model}_error"
        if error_col not in daily_df.columns:
            continue
        errors = daily_df[error_col].dropna()
        ax.hist(
            errors, bins=30, color=PLOT_COLORS[model], alpha=0.7, edgecolor="white",
        )
        ax.axvline(0, color="black", linestyle="--", alpha=0.5)
        ax.axvline(errors.mean(), color="red", linestyle="-", alpha=0.8, label=f"Biais: {errors.mean():.2f}°C")
        ax.set_xlabel("Erreur (°C)")
        ax.set_ylabel("Fréquence")
        ax.set_title(f"Distribution des erreurs — {model.upper()}")
        ax.legend()

    fig.suptitle("Distribution des erreurs de prévision", fontsize=14)
    _save_fig(fig, "07_error_distribution")


def plot_seasonal_heatmap(daily_df: pd.DataFrame) -> None:
    """Plot 8: Heatmap of bucket accuracy by day of week × month."""
    _setup_style()

    from src.backtest.buckets import assign_bucket, generate_daily_buckets

    df = daily_df.copy()
    df["month"] = df["date"].dt.month
    df["dow"] = df["date"].dt.dayofweek  # 0=Mon, 6=Sun

    # Use GFS for this heatmap (most common model)
    model = "gfs"
    forecast_col = f"{model}_forecast"
    if forecast_col not in df.columns:
        return

    # Compute bucket accuracy per (dow, month)
    records = []
    for _, row in df.iterrows():
        forecast = row[forecast_col]
        actual = row["actual_temp"]
        if pd.isna(forecast) or pd.isna(actual):
            continue
        buckets = generate_daily_buckets(row["month"])
        pred_b = assign_bucket(forecast, buckets)
        act_b = assign_bucket(actual, buckets)
        if pred_b and act_b:
            records.append({
                "month": row["month"],
                "dow": row["dow"],
                "correct": pred_b["index"] == act_b["index"],
            })

    if not records:
        return

    rec_df = pd.DataFrame(records)
    pivot = rec_df.groupby(["dow", "month"])["correct"].mean().unstack(fill_value=0) * 100

    fig, ax = plt.subplots(figsize=(12, 5))
    month_labels = ["Jan", "Fév", "Mar", "Avr", "Mai", "Jun",
                     "Jul", "Aoû", "Sep", "Oct", "Nov", "Déc"]
    dow_labels = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]

    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([month_labels[m - 1] for m in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([dow_labels[d] for d in pivot.index])
    ax.set_title("Bucket Accuracy GFS — Jour de la semaine × Mois (%)")
    fig.colorbar(im, ax=ax, label="Accuracy (%)")

    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.values[i, j]:.0f}", ha="center", va="center", fontsize=8)

    _save_fig(fig, "08_seasonal_heatmap")


def plot_model_bias(model_metrics: dict) -> None:
    """Plot 9: Monthly bias bar chart for each model."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(12, 5))

    months = sorted(set().union(*(m.get("monthly_bias", {}).keys() for m in model_metrics.values())))
    x = np.arange(len(months))
    width = 0.35

    for i, model in enumerate(["gfs", "ecmwf"]):
        if model not in model_metrics:
            continue
        bias_vals = [model_metrics[model]["monthly_bias"].get(m, 0) for m in months]
        ax.bar(
            x + i * width, bias_vals, width,
            label=model.upper(), color=PLOT_COLORS[model],
        )

    ax.axhline(0, color="black", linestyle="-", alpha=0.3)
    month_labels = ["Jan", "Fév", "Mar", "Avr", "Mai", "Jun",
                     "Jul", "Aoû", "Sep", "Oct", "Nov", "Déc"]
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([month_labels[m - 1] for m in months])
    ax.set_ylabel("Biais moyen (°C)")
    ax.set_title("Biais mensuel des modèles (+ = surestimation, - = sous-estimation)")
    ax.legend()
    _save_fig(fig, "09_model_bias")


def plot_lead_time_analysis(trades_df: pd.DataFrame, strategy_summaries: dict) -> None:
    """Plot 10: Lead time sweet spot — win rate, P/L per trade, and total P/L by entry timing."""
    _setup_style()

    # Collect early_conv strategies + original convergence
    entries = []
    for strat_name, s in strategy_summaries.items():
        if strat_name.startswith("early_conv_"):
            hours = int(strat_name.split("_")[-1].replace("h", ""))
            entries.append({
                "lead_hours": hours,
                "label": f"J-{'2' if hours > 24 else '1'} soir\n({hours}h)",
                "win_rate": s["win_rate"] * 100,
                "pnl_per_trade": s["total_pnl"] / s["total_trades"] if s["total_trades"] > 0 else 0,
                "total_pnl": s["total_pnl"],
                "total_trades": s["total_trades"],
                "avg_price": trades_df[
                    trades_df["strategy"] == f"early_conv_{hours}h"
                ]["price"].mean() if len(trades_df[trades_df["strategy"] == f"early_conv_{hours}h"]) > 0 else 0,
            })

    # Add original convergence as reference (18h, fixed pricing)
    if "convergence" in strategy_summaries:
        s = strategy_summaries["convergence"]
        conv_trades = trades_df[trades_df["strategy"] == "convergence"]
        entries.append({
            "lead_hours": 18,
            "label": "Conv. fixe\n(ref 18h)",
            "win_rate": s["win_rate"] * 100,
            "pnl_per_trade": s["total_pnl"] / s["total_trades"] if s["total_trades"] > 0 else 0,
            "total_pnl": s["total_pnl"],
            "total_trades": s["total_trades"],
            "avg_price": conv_trades["price"].mean() if len(conv_trades) > 0 else 0,
        })

    if not entries:
        return

    entries.sort(key=lambda e: e["lead_hours"], reverse=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    labels = [e["label"] for e in entries]
    x = range(len(entries))

    # Panel 1: Win rate + buy price
    ax1 = axes[0]
    bars = ax1.bar(x, [e["win_rate"] for e in entries], color=["#FF9800" if "ref" not in e["label"] else "#9E9E9E" for e in entries])
    ax1.set_ylabel("Win Rate (%)")
    ax1.set_title("Win Rate par timing d'entrée")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, fontsize=9)
    for i, e in enumerate(entries):
        ax1.text(i, e["win_rate"] + 1, f"{e['win_rate']:.1f}%", ha="center", fontsize=10, fontweight="bold")
        ax1.text(i, e["win_rate"] / 2, f"prix: {e['avg_price']:.0%}", ha="center", fontsize=9, color="white", fontweight="bold")

    # Panel 2: P/L per trade (the real alpha metric)
    ax2 = axes[1]
    pnl_colors = ["#4CAF50" if e["pnl_per_trade"] > 0 else "#F44336" for e in entries]
    ax2.bar(x, [e["pnl_per_trade"] for e in entries], color=pnl_colors)
    ax2.set_ylabel("P/L moyen par trade ($)")
    ax2.set_title("Alpha par trade selon le timing")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.axhline(0, color="black", linestyle="-", alpha=0.3)
    for i, e in enumerate(entries):
        offset = 0.01 if e["pnl_per_trade"] >= 0 else -0.03
        ax2.text(i, e["pnl_per_trade"] + offset, f"${e['pnl_per_trade']:.3f}", ha="center", fontsize=10, fontweight="bold")

    # Panel 3: Total P/L
    ax3 = axes[2]
    pnl_colors = ["#4CAF50" if e["total_pnl"] > 0 else "#F44336" for e in entries]
    ax3.bar(x, [e["total_pnl"] for e in entries], color=pnl_colors)
    ax3.set_ylabel("P/L total ($)")
    ax3.set_title("P/L total sur la période")
    ax3.set_xticks(list(x))
    ax3.set_xticklabels(labels, fontsize=9)
    ax3.axhline(0, color="black", linestyle="-", alpha=0.3)
    for i, e in enumerate(entries):
        offset = 2 if e["total_pnl"] >= 0 else -8
        ax3.text(i, e["total_pnl"] + offset, f"${e['total_pnl']:.0f}\n({e['total_trades']} trades)", ha="center", fontsize=9, fontweight="bold")

    fig.suptitle("Analyse du sweet spot : timing d'entrée vs alpha", fontsize=14, fontweight="bold")
    _save_fig(fig, "10_lead_time_sweetspot")


def plot_early_vs_late_cumulative(trades_df: pd.DataFrame) -> None:
    """Plot 11: Cumulative P/L comparison between early and late entry strategies."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(14, 6))

    strategy_styles = {
        "early_conv_36h": {"color": "#E91E63", "label": "Early entry J-2 (36h, prix bas)"},
        "early_conv_18h": {"color": "#FF9800", "label": "Early entry J-1 (18h, prix moyen)"},
        "convergence": {"color": "#9C27B0", "label": "Convergence fixe (ref)"},
    }

    for strat, style in strategy_styles.items():
        strat_trades = trades_df[trades_df["strategy"] == strat].sort_values("date")
        if strat_trades.empty:
            continue
        cumulative = strat_trades["pnl"].cumsum()
        ax.plot(strat_trades["date"].values, cumulative.values,
                label=style["label"], color=style["color"], linewidth=2)

    ax.axhline(0, color="black", linestyle="-", alpha=0.3)
    ax.set_xlabel("Date")
    ax.set_ylabel("P/L cumulé ($)")
    ax.set_title("P/L cumulé : entrée précoce (J-2) vs tardive (J-1)")
    ax.legend()
    _save_fig(fig, "11_early_vs_late_pnl")


def generate_all_plots(
    daily_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    model_metrics: dict,
    convergence_stats: dict,
    strategy_summaries: dict | None = None,
) -> None:
    """Generate all plots.

    Args:
        daily_df: Daily forecasts + actuals.
        trades_df: Trades log.
        model_metrics: Dict from compute_all_metrics.
        convergence_stats: Dict from compute_convergence_analysis.
        strategy_summaries: Dict from compute_strategy_summary (for lead time analysis).
    """
    logger.info("Generating plots...")
    plot_forecast_vs_actual(daily_df)
    plot_error_timeseries(daily_df)
    plot_monthly_mae(model_metrics)
    plot_bucket_accuracy_by_month(model_metrics)
    plot_cumulative_pnl(trades_df)
    plot_convergence_analysis(convergence_stats)
    plot_error_distribution(daily_df)
    plot_seasonal_heatmap(daily_df)
    plot_model_bias(model_metrics)
    if strategy_summaries is not None:
        plot_lead_time_analysis(trades_df, strategy_summaries)
        plot_early_vs_late_cumulative(trades_df)
    logger.info("All plots generated.")
