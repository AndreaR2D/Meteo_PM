"""Main entry point for the London weather backtest."""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DATA_PROCESSED_DIR,
    DEFAULT_END_DATE,
    DEFAULT_PRICE_SCENARIO,
    DEFAULT_START_DATE,
    OUTPUT_REPORTS_DIR,
)
from src.data.fetch_forecasts import fetch_all_forecasts
from src.data.fetch_actuals import fetch_actuals
from src.backtest.engine import build_daily_dataframe, run_backtest, save_processed_data
from src.analysis.model_accuracy import compute_all_metrics
from src.analysis.pnl import (
    compute_convergence_analysis,
    compute_monthly_summary,
    compute_strategy_summary,
)
from src.analysis.plots import generate_all_plots

logger = logging.getLogger(__name__)


def generate_report(
    start_date: str,
    end_date: str,
    model_metrics: dict,
    strategy_summaries: dict,
    convergence_stats: dict,
) -> str:
    """Generate a markdown backtest report.

    Args:
        start_date: Backtest start date.
        end_date: Backtest end date.
        model_metrics: Dict from compute_all_metrics.
        strategy_summaries: Dict from compute_strategy_summary.
        convergence_stats: Dict from compute_convergence_analysis.

    Returns:
        Markdown report string.
    """
    # Determine best model
    best_model = min(model_metrics, key=lambda m: model_metrics[m]["mae"])
    best_mae = model_metrics[best_model]["mae"]
    best_acc = model_metrics[best_model]["bucket_accuracy"] * 100

    # Best strategy by P/L
    best_strategy = max(strategy_summaries, key=lambda s: strategy_summaries[s]["total_pnl"])
    best_pnl = strategy_summaries[best_strategy]["total_pnl"]

    conv_wr = convergence_stats["convergence_win_rate"] * 100

    lines = [
        f"# Backtest Report — London Weather Trading",
        f"",
        f"## Période : {start_date} → {end_date}",
        f"",
        f"## Résumé exécutif",
        f"",
        f"- **Meilleur modèle** : {best_model.upper()} (MAE: {best_mae:.2f}°C, Bucket Accuracy: {best_acc:.1f}%)",
        f"- **Win rate stratégie convergence** : {conv_wr:.1f}%",
        f"- **Meilleure stratégie P/L** : {best_strategy} (${best_pnl:+.2f})",
        f"- **Jours de convergence** : {convergence_stats['convergence_days']} / {convergence_stats['convergence_days'] + convergence_stats['divergence_days']}",
        f"",
        f"---",
        f"",
        f"## Détail par modèle",
        f"",
    ]

    for model in ["gfs", "ecmwf"]:
        if model not in model_metrics:
            continue
        m = model_metrics[model]
        lines.extend([
            f"### {model.upper()}",
            f"- Jours valides : {m['valid_days']}",
            f"- MAE globale : {m['mae']:.2f}°C",
            f"- Biais moyen : {m['bias']:+.2f}°C",
            f"- Bucket accuracy : {m['bucket_accuracy'] * 100:.1f}%",
            f"",
        ])

    lines.extend([
        f"---",
        f"",
        f"## Détail par stratégie",
        f"",
    ])

    for strategy, s in strategy_summaries.items():
        lines.extend([
            f"### {strategy.replace('_', ' ').title()}",
            f"- Trades : {s['total_trades']}",
            f"- Wins : {s['wins']} ({s['win_rate'] * 100:.1f}%)",
            f"- P/L total : ${s['total_pnl']:+.2f}",
            f"- Max drawdown : ${s['max_drawdown']:.2f}",
            f"- Sharpe ratio : {s['sharpe_ratio']:.2f}",
            f"- Meilleur mois : {s['best_month']} (${s['best_month_pnl']:+.2f})",
            f"- Pire mois : {s['worst_month']} (${s['worst_month_pnl']:+.2f})",
            f"",
        ])

    lines.extend([
        f"---",
        f"",
        f"## Analyse convergence",
        f"",
        f"| Condition | Jours | Wins | Win Rate |",
        f"|-----------|-------|------|----------|",
        f"| Convergence | {convergence_stats['convergence_days']} | {convergence_stats['convergence_wins']} | {convergence_stats['convergence_win_rate'] * 100:.1f}% |",
        f"| Divergence | {convergence_stats['divergence_days']} | {convergence_stats['divergence_wins']} | {convergence_stats['divergence_win_rate'] * 100:.1f}% |",
        f"",
        f"---",
        f"",
        f"## Analyse saisonnière",
        f"",
        f"### MAE par mois",
        f"",
        f"| Mois | GFS MAE | ECMWF MAE |",
        f"|------|---------|-----------|",
    ])

    month_names = ["Jan", "Fév", "Mar", "Avr", "Mai", "Jun",
                   "Jul", "Aoû", "Sep", "Oct", "Nov", "Déc"]

    all_months = sorted(set().union(
        *(m.get("monthly_mae", {}).keys() for m in model_metrics.values())
    ))
    for month in all_months:
        gfs_mae = model_metrics.get("gfs", {}).get("monthly_mae", {}).get(month, "-")
        ecmwf_mae = model_metrics.get("ecmwf", {}).get("monthly_mae", {}).get(month, "-")
        gfs_str = f"{gfs_mae:.2f}" if isinstance(gfs_mae, float) else gfs_mae
        ecmwf_str = f"{ecmwf_mae:.2f}" if isinstance(ecmwf_mae, float) else ecmwf_mae
        lines.append(f"| {month_names[month - 1]} | {gfs_str} | {ecmwf_str} |")

    lines.extend([
        f"",
        f"---",
        f"",
        f"## Recommandations",
        f"",
        f"1. Le modèle **{best_model.upper()}** est globalement plus précis sur Londres",
        f"2. La stratégie de **convergence** {'surperforme' if conv_wr > 50 else 'ne surperforme pas'} avec un win rate de {conv_wr:.1f}%",
        f"3. Les mois avec la plus grande erreur méritent une attention particulière pour le sizing des positions",
        f"4. Considérer l'intégration des prix Polymarket réels (V2) pour un backtest plus réaliste",
        f"",
    ])

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="London Weather Prediction Backtest"
    )
    parser.add_argument(
        "--start-date", default=DEFAULT_START_DATE,
        help=f"Start date (default: {DEFAULT_START_DATE})",
    )
    parser.add_argument(
        "--end-date", default=DEFAULT_END_DATE,
        help=f"End date (default: {DEFAULT_END_DATE})",
    )
    parser.add_argument(
        "--strategy", nargs="+", default=None,
        help="Strategies to run (default: all). Choices: naive_gfs, naive_ecmwf, best_model, convergence",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip plot generation",
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="Re-download data even if cached",
    )
    parser.add_argument(
        "--scenario", default=DEFAULT_PRICE_SCENARIO,
        choices=["efficient_market", "inefficient_market"],
        help=f"Price scenario (default: {DEFAULT_PRICE_SCENARIO})",
    )
    parser.add_argument(
        "--temp-source", default="archive",
        choices=["archive", "best_match"],
        help="Temperature source: 'archive' (observations, recommended) or 'best_match' (legacy)",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full backtest pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()

    # --- Step 1: Fetch data ---
    logger.info("=" * 60)
    logger.info("LONDON WEATHER BACKTEST")
    logger.info("Period: %s → %s", args.start_date, args.end_date)
    logger.info("=" * 60)

    logger.info("Fetching J-1 forecast data...")
    forecasts_j1 = fetch_all_forecasts(
        args.start_date, args.end_date, lead_time=1, refresh=args.refresh
    )

    logger.info("Fetching J-2 forecast data...")
    forecasts_j2 = fetch_all_forecasts(
        args.start_date, args.end_date, lead_time=2, refresh=args.refresh
    )

    logger.info("Fetching actual temperatures (source: %s)...", args.temp_source)
    actuals = fetch_actuals(
        args.start_date, args.end_date, refresh=args.refresh, source=args.temp_source
    )

    # --- Step 2: Build daily DataFrame ---
    logger.info("Building daily dataset...")
    daily_df = build_daily_dataframe(forecasts_j1, actuals, forecasts_j2=forecasts_j2)
    logger.info("Daily dataset: %d rows", len(daily_df))

    if daily_df.empty:
        logger.error("No data available. Exiting.")
        sys.exit(1)

    # --- Step 3: Run backtest ---
    logger.info("Running backtest...")
    trades_df = run_backtest(daily_df, strategies=args.strategy, scenario=args.scenario)

    # --- Step 4: Compute metrics ---
    logger.info("Computing model accuracy metrics...")
    model_metrics = compute_all_metrics(daily_df)

    logger.info("Computing strategy performance...")
    strategy_summaries = compute_strategy_summary(trades_df)
    convergence_stats = compute_convergence_analysis(trades_df)

    # --- Step 5: Save processed data ---
    logger.info("Saving processed data...")
    save_processed_data(daily_df, trades_df)

    monthly_summary = compute_monthly_summary(daily_df, trades_df, model_metrics)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    monthly_summary.to_csv(DATA_PROCESSED_DIR / "monthly_summary.csv", index=False)

    # --- Step 6: Generate plots ---
    if not args.no_plots:
        generate_all_plots(daily_df, trades_df, model_metrics, convergence_stats, strategy_summaries)

    # --- Step 7: Generate report ---
    logger.info("Generating report...")
    report = generate_report(
        args.start_date, args.end_date,
        model_metrics, strategy_summaries, convergence_stats,
    )
    OUTPUT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_REPORTS_DIR / "backtest_report.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info("Report saved to %s", report_path)

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("BACKTEST COMPLETE")
    print("=" * 60)

    for model in ["gfs", "ecmwf"]:
        if model in model_metrics:
            m = model_metrics[model]
            print(f"\n{model.upper()}:")
            print(f"  MAE: {m['mae']:.2f}°C | Bias: {m['bias']:+.2f}°C | Bucket Accuracy: {m['bucket_accuracy'] * 100:.1f}%")

    print(f"\nStrategies:")
    for strategy, s in strategy_summaries.items():
        print(f"  {strategy}: Win Rate {s['win_rate'] * 100:.1f}% | P/L: ${s['total_pnl']:+.2f} | Trades: {s['total_trades']}")

    print(f"\nConvergence: {convergence_stats['convergence_win_rate'] * 100:.1f}% win rate "
          f"({convergence_stats['convergence_days']} days)")
    print(f"Divergence:  {convergence_stats['divergence_win_rate'] * 100:.1f}% win rate "
          f"({convergence_stats['divergence_days']} days)")

    # Survivorship bias report
    forced = {k: v for k, v in strategy_summaries.items() if k == "forced_bet"}
    conv = {k: v for k, v in strategy_summaries.items() if k == "convergence"}
    if forced and conv:
        f = forced["forced_bet"]
        c = conv["convergence"]
        bias = c["win_rate"] * 100 - f["win_rate"] * 100
        print(f"\n--- SURVIVORSHIP BIAS CHECK ---")
        print(f"  Convergence only: {c['win_rate']*100:.1f}% WR ({c['total_trades']} trades)")
        print(f"  Forced bet (all): {f['win_rate']*100:.1f}% WR ({f['total_trades']} trades)")
        print(f"  Bias: {bias:+.1f}pp (convergence inflates by this much)")
        # Breakdown: convergence vs divergence days in forced_bet
        forced_trades = trades_df[trades_df["strategy"].str.startswith("forced_bet")]
        if "converge" in forced_trades.columns:
            conv_days = forced_trades[forced_trades["converge"] == True]
            div_days = forced_trades[forced_trades["converge"] == False]
            if len(conv_days) > 0 and len(div_days) > 0:
                print(f"  Forced bet on convergence days: {conv_days['win'].mean()*100:.1f}% WR ({len(conv_days)} trades)")
                print(f"  Forced bet on divergence days:  {div_days['win'].mean()*100:.1f}% WR ({len(div_days)} trades)")

    # Sweet spot analysis
    early_strats = {k: v for k, v in strategy_summaries.items() if k.startswith("early_conv_")}
    if early_strats:
        print(f"\n--- SWEET SPOT ANALYSIS (timing vs alpha) ---")
        for name, s in sorted(early_strats.items(), key=lambda x: -int(x[0].split("_")[-1].replace("h", ""))):
            pnl_per_trade = s["total_pnl"] / s["total_trades"] if s["total_trades"] > 0 else 0
            print(f"  {name}: Win Rate {s['win_rate']*100:.1f}% | "
                  f"P/L/trade: ${pnl_per_trade:.3f} | "
                  f"Total P/L: ${s['total_pnl']:+.2f} | "
                  f"Trades: {s['total_trades']}")

    print(f"\nReport: {report_path}")
    print(f"Plots:  {OUTPUT_REPORTS_DIR.parent / 'plots'}/")


if __name__ == "__main__":
    main()
