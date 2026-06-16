"""
visualize.py
--------------
Generates all project visualizations and saves them to the
visualizations/ directory. Covers:

  1.  Demand series overview (historical trend)
  2.  Seasonal decomposition (trend / seasonal / residual)
  3.  Seasonality bar chart (average effect by month)
  4.  Prophet forecast vs actuals
  5.  SARIMAX forecast with confidence intervals
  6.  LSTM forecast (if available)
  7.  All-model forecast comparison
  8.  Scenario analysis (best / most-likely / worst)
  9.  Prophet backtest (actual vs predicted)
  10. SARIMAX backtest (actual vs predicted)
  11. LSTM backtest (if available)
  12. All-model backtest comparison
  13. Backtest metrics comparison (bar chart)
  14. Top skills bar chart
  15. Emerging roles bar chart
  16. Top industries bar chart
  17. Monthly skill trends heatmap

Run standalone:
    python analysis/visualize.py

Or called from main.py's step_visualize().
"""

import os
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Style ──────────────────────────────────────────────────────────────────────
PALETTE = {
    "actual":      "#2C3E50",
    "prophet":     "#2980B9",
    "sarimax":     "#27AE60",
    "lstm":        "#E67E22",
    "best":        "#27AE60",
    "most_likely": "#2980B9",
    "worst":       "#E74C3C",
    "ci":          "#AED6F1",
    "grid":        "#ECF0F1",
    "accent":      "#8E44AD",
}

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.grid":         True,
    "grid.color":        PALETTE["grid"],
    "grid.linewidth":    0.8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "DejaVu Sans",
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "legend.framealpha": 0.9,
})

# ── Helpers ────────────────────────────────────────────────────────────────────


def _save(fig, path, tight=True):
    if tight:
        plt.tight_layout()

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def _fmt_date_axis(ax, freq="ME"):
    """Apply clean date formatting to x-axis."""

    locator = mdates.MonthLocator(interval=3)
    formatter = mdates.DateFormatter("%b %Y")
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")


def _load(path, **kwargs):
    """Load CSV, return empty DataFrame on missing file."""

    try:
        return pd.read_csv(path, **kwargs)
    except FileNotFoundError:
        logger.warning("File not found, skipping: %s", path)
        return pd.DataFrame()


# ── 1. Demand series overview ──────────────────────────────────────────────────

def plot_demand_overview(monthly_series, out_dir):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(monthly_series.index, monthly_series.values,
            color=PALETTE["actual"], linewidth=2, label="Demand index")
    ax.fill_between(monthly_series.index, monthly_series.values,
                    alpha=0.15, color=PALETTE["actual"])
    ax.set_title("Historical Market Demand Index (Google Trends composite)")
    ax.set_ylabel("Demand index (0–100 scale)")
    ax.set_xlabel("")
    _fmt_date_axis(ax)
    ax.legend()
    _save(fig, os.path.join(out_dir, "01_demand_overview.png"))


# ── 2. Seasonal decomposition ──────────────────────────────────────────────────

def plot_decomposition(monthly_series, out_dir):
    from statsmodels.tsa.seasonal import seasonal_decompose

    series = monthly_series.asfreq("ME").interpolate()
    if len(series) < 24:
        logger.warning(
            "Too few data points for seasonal decomposition, skipping.")
        return
    result = seasonal_decompose(series, model="additive", period=12)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    components = [
        ("Observed",  result.observed,  PALETTE["actual"]),
        ("Trend",     result.trend,     PALETTE["prophet"]),
        ("Seasonal",  result.seasonal,  PALETTE["sarimax"]),
        ("Residual",  result.resid,     PALETTE["accent"]),
    ]
    for ax, (label, data, color) in zip(axes, components):
        ax.plot(data.index, data.values, color=color, linewidth=1.6)
        ax.set_ylabel(label)
        ax.grid(True, color=PALETTE["grid"])

    axes[0].set_title("Seasonal Decomposition of Demand Index")
    _fmt_date_axis(axes[-1])
    _save(fig, os.path.join(out_dir, "02_seasonal_decomposition.png"))


# ── 3. Seasonality by month ────────────────────────────────────────────────────

def plot_seasonality_by_month(seasonality_df, out_dir):
    if seasonality_df.empty:
        return

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    # seasonality_df may have 'month' as index or column
    if "seasonal" in seasonality_df.columns:
        s = seasonality_df.set_index(seasonality_df.columns[0])["seasonal"]
    else:
        s = seasonality_df.iloc[:, 0]

    s.index = [month_names[int(i)-1] for i in s.index]
    colors = [PALETTE["sarimax"] if v >= 0 else PALETTE["worst"]
              for v in s.values]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(s.index, s.values, color=colors,
                  edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="#555", linewidth=0.8, linestyle="--")
    ax.set_title("Average Seasonal Effect by Calendar Month")
    ax.set_ylabel("Seasonal component")
    ax.set_xlabel("Month")
    for bar, val in zip(bars, s.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    _save(fig, os.path.join(out_dir, "03_seasonality_by_month.png"))


# ── 4. Prophet forecast ────────────────────────────────────────────────────────

def plot_prophet_forecast(prophet_df, monthly_series, out_dir):
    if prophet_df.empty:
        return

    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
    cutoff = monthly_series.index.max()
    hist = prophet_df[prophet_df["ds"] <= cutoff]
    future = prophet_df[prophet_df["ds"] > cutoff]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(monthly_series.index, monthly_series.values,
            color=PALETTE["actual"], linewidth=2, label="Actual", zorder=3)
    ax.plot(hist["ds"], hist["yhat"],
            color=PALETTE["prophet"], linewidth=1.5, linestyle="--",
            alpha=0.7, label="Prophet fitted")
    ax.fill_between(hist["ds"], hist["yhat_lower"], hist["yhat_upper"],
                    alpha=0.15, color=PALETTE["prophet"])
    ax.plot(future["ds"], future["yhat"],
            color=PALETTE["prophet"], linewidth=2.5, label="Prophet forecast")
    ax.fill_between(future["ds"], future["yhat_lower"], future["yhat_upper"],
                    alpha=0.25, color=PALETTE["prophet"], label="95% CI")
    ax.axvline(cutoff, color="#888", linewidth=1,
               linestyle=":", label="Forecast start")
    ax.set_title("Prophet: Demand Forecast (12 months)")
    ax.set_ylabel("Demand index")
    _fmt_date_axis(ax)
    ax.legend()
    _save(fig, os.path.join(out_dir, "04_prophet_forecast.png"))


# ── 5. SARIMAX forecast ────────────────────────────────────────────────────────

def plot_sarimax_forecast(sarimax_df, monthly_series, out_dir):
    if sarimax_df.empty:
        return

    sarimax_df.index = pd.to_datetime(sarimax_df.index)
    cutoff = monthly_series.index.max()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(monthly_series.index, monthly_series.values,
            color=PALETTE["actual"], linewidth=2, label="Actual", zorder=3)
    ax.plot(sarimax_df.index, sarimax_df["forecast"],
            color=PALETTE["sarimax"], linewidth=2.5, label="SARIMAX forecast")
    lower_col = [c for c in sarimax_df.columns if "lower" in c]
    upper_col = [c for c in sarimax_df.columns if "upper" in c]

    if lower_col and upper_col:
        ax.fill_between(sarimax_df.index,
                        sarimax_df[lower_col[0]], sarimax_df[upper_col[0]],
                        alpha=0.25, color=PALETTE["sarimax"], label="95% CI")

    ax.axvline(cutoff, color="#888", linewidth=1,
               linestyle=":", label="Forecast start")
    ax.set_title(
        "SARIMAX: Demand Forecast with Confidence Intervals (12 months)")
    ax.set_ylabel("Demand index")
    _fmt_date_axis(ax)
    ax.legend()
    _save(fig, os.path.join(out_dir, "05_sarimax_forecast.png"))


# ── 6. LSTM forecast ───────────────────────────────────────────────────────────

def plot_lstm_forecast(lstm_df, monthly_series, out_dir):
    if lstm_df.empty or lstm_df["yhat"].isna().all():
        logger.info("No valid LSTM forecast to plot.")
        return

    lstm_df["ds"] = pd.to_datetime(lstm_df["ds"])
    cutoff = monthly_series.index.max()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(monthly_series.index, monthly_series.values,
            color=PALETTE["actual"], linewidth=2, label="Actual", zorder=3)
    ax.plot(lstm_df["ds"], lstm_df["yhat"],
            color=PALETTE["lstm"], linewidth=2.5, label="LSTM forecast")
    ax.axvline(cutoff, color="#888", linewidth=1,
               linestyle=":", label="Forecast start")
    ax.set_title("LSTM: Demand Forecast (12 months)")
    ax.set_ylabel("Demand index")
    _fmt_date_axis(ax)
    ax.legend()
    _save(fig, os.path.join(out_dir, "06_lstm_forecast.png"))


# ── 7. All-model forecast comparison ──────────────────────────────────────────

def plot_forecast_comparison(prophet_df, sarimax_df, lstm_df, monthly_series, out_dir):
    cutoff = monthly_series.index.max()

    fig, ax = plt.subplots(figsize=(13, 5))
    # Historical
    ax.plot(monthly_series.index, monthly_series.values,
            color=PALETTE["actual"], linewidth=2.5, label="Actual", zorder=5)
    ax.axvline(cutoff, color="#888", linewidth=1,
               linestyle=":", label="Forecast start")

    if not prophet_df.empty:
        prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
        future_p = prophet_df[prophet_df["ds"] > cutoff]
        ax.plot(future_p["ds"], future_p["yhat"],
                color=PALETTE["prophet"], linewidth=2, label="Prophet", linestyle="--")

    if not sarimax_df.empty:
        sarimax_df.index = pd.to_datetime(sarimax_df.index)
        ax.plot(sarimax_df.index, sarimax_df["forecast"],
                color=PALETTE["sarimax"], linewidth=2, label="SARIMAX", linestyle="-.")

    if not lstm_df.empty and not lstm_df["yhat"].isna().all():
        lstm_df["ds"] = pd.to_datetime(lstm_df["ds"])
        ax.plot(lstm_df["ds"], lstm_df["yhat"],
                color=PALETTE["lstm"], linewidth=2, label="LSTM", linestyle=":")

    ax.set_title("Model Comparison: 12-Month Demand Forecasts")
    ax.set_ylabel("Demand index")
    _fmt_date_axis(ax)
    ax.legend()
    _save(fig, os.path.join(out_dir, "07_forecast_comparison.png"))


# ── 8. Scenario analysis ───────────────────────────────────────────────────────

def plot_scenarios(scenario_df, monthly_series, out_dir):
    if scenario_df.empty:
        return

    scenario_df.index = pd.to_datetime(scenario_df.index)
    cutoff = monthly_series.index.max()
    future = scenario_df[scenario_df.index > cutoff]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(monthly_series.index, monthly_series.values,
            color=PALETTE["actual"], linewidth=2, label="Actual", zorder=5)
    ax.axvline(cutoff, color="#888", linewidth=1,
               linestyle=":", label="Forecast start")

    col_map = {
        "best_case":   (PALETTE["best"],        "Best case"),
        "most_likely": (PALETTE["most_likely"], "Most likely"),
        "worst_case":  (PALETTE["worst"],       "Worst case"),
    }
    for col, (color, label) in col_map.items():
        if col in future.columns:
            ax.plot(future.index, future[col],
                    color=color, linewidth=2.5, label=label)

    if "best_case" in future.columns and "worst_case" in future.columns:
        ax.fill_between(future.index, future["best_case"], future["worst_case"],
                        alpha=0.12, color=PALETTE["most_likely"], label="Scenario range")

    ax.set_title(
        "Scenario Analysis: Best / Most Likely / Worst Case (12 months)")
    ax.set_ylabel("Demand index")
    _fmt_date_axis(ax)
    ax.legend()
    _save(fig, os.path.join(out_dir, "08_scenario_analysis.png"))


# ── 9-11. Backtests ────────────────────────────────────────────────────────────

def _plot_single_backtest(backtest_df, model_name, color, metrics, ax):
    backtest_df["ds"] = pd.to_datetime(backtest_df["ds"])
    ax.plot(backtest_df["ds"], backtest_df["actual"],
            color=PALETTE["actual"], linewidth=2, marker="o",
            markersize=5, label="Actual", zorder=3)
    ax.plot(backtest_df["ds"], backtest_df["predicted"],
            color=color, linewidth=2, marker="s",
            markersize=5, linestyle="--", label=f"{model_name} predicted")

    if metrics:
        info = f"MAE={metrics.get('MAE',0):.2f}  RMSE={metrics.get('RMSE',0):.2f}  MAPE={metrics.get('MAPE_%',0):.1f}%"
        ax.set_title(f"{model_name} Backtest (6-month holdout)  |  {info}")
    else:
        ax.set_title(f"{model_name} Backtest (6-month holdout)")

    ax.set_ylabel("Demand index")
    _fmt_date_axis(ax)
    ax.legend()


def plot_backtest(backtest_df, metrics, model_name, color, out_dir, fname):
    if backtest_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    _plot_single_backtest(backtest_df, model_name, color, metrics, ax)
    _save(fig, os.path.join(out_dir, fname))


# ── 12. All-model backtest comparison ─────────────────────────────────────────

def plot_backtest_comparison(backtests, out_dir):
    """
    backtests: list of (df, model_name, color, metrics) tuples, non-empty only
    """

    valid = [(df, name, col, met)
             for df, name, col, met in backtests if not df.empty]

    if not valid:
        return

    n = len(valid)
    fig, axes = plt.subplots(n, 1, figsize=(11, 4*n), sharex=False)

    if n == 1:
        axes = [axes]

    for ax, (df, name, color, metrics) in zip(axes, valid):
        _plot_single_backtest(df, name, color, metrics, ax)

    fig.suptitle(
        "Model Backtests: Actual vs Predicted (6-month holdout)", fontsize=14, y=1.01)
    _save(fig, os.path.join(out_dir, "12_backtest_comparison.png"))


# ── 13. Metrics comparison bar chart ──────────────────────────────────────────

def plot_metrics_comparison(metrics_dict, out_dir):
    """
    metrics_dict: { model_name: { 'MAE': float, 'RMSE': float, 'MAPE_%': float } }
    """

    valid = {k: v for k, v in metrics_dict.items() if v}
    if not valid:
        return

    models = list(valid.keys())
    x = np.arange(len(models))
    width = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    metric_labels = [("MAE", "Mean Absolute Error"),
                     ("RMSE", "Root Mean Squared Error"),
                     ("MAPE_%", "MAPE (%)")]
    colors = [PALETTE["prophet"], PALETTE["sarimax"], PALETTE["lstm"]]

    for ax, (metric_key, metric_title), color in zip(axes, metric_labels, colors):
        vals = [valid[m].get(metric_key, 0) for m in models]
        bars = ax.bar(models, vals, color=colors[:len(
            models)], edgecolor="white", width=0.5)
        ax.set_title(metric_title)
        ax.set_ylabel(metric_key.replace("_", " "))
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_ylim(0, max(vals)*1.3 if vals else 1)

    fig.suptitle("Backtest Accuracy Metrics by Model", fontsize=14)
    _save(fig, os.path.join(out_dir, "13_metrics_comparison.png"))


# ── 14. Top skills ─────────────────────────────────────────────────────────────

def plot_top_skills(skills_path, out_dir, top_n=15):
    df = _load(skills_path, header=None, names=["skill", "count"])
    if df.empty:
        return

    df["count"] = pd.to_numeric(df["count"], errors="coerce")
    df = df.dropna().sort_values("count", ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(df["skill"], df["count"],
                   color=PALETTE["prophet"], edgecolor="white")
    ax.set_title(f"Top {top_n} In-Demand Skills (from job postings)")
    ax.set_xlabel("Job posting mentions")

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.2, bar.get_y() + bar.get_height()/2,
                str(int(w)), va="center", fontsize=8)

    _save(fig, os.path.join(out_dir, "14_top_skills.png"))


# ── 15. Emerging roles ─────────────────────────────────────────────────────────

def plot_emerging_roles(roles_path, out_dir, top_n=15):
    df = _load(roles_path, header=None, names=["role", "count"])
    if df.empty:
        return

    df["count"] = pd.to_numeric(df["count"], errors="coerce")
    df = df.dropna().sort_values("count", ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(df["role"], df["count"],
                   color=PALETTE["sarimax"], edgecolor="white")
    ax.set_title(f"Top {top_n} Emerging Job Roles (by posting volume)")
    ax.set_xlabel("Job posting count")

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.05, bar.get_y() + bar.get_height()/2,
                str(int(w)), va="center", fontsize=8)

    _save(fig, os.path.join(out_dir, "15_emerging_roles.png"))


# ── 16. Top industries ─────────────────────────────────────────────────────────

def plot_top_industries(industries_path, out_dir):
    df = _load(industries_path, header=None, names=["industry", "count"])
    if df.empty:
        return

    df["count"] = pd.to_numeric(df["count"], errors="coerce")
    df = df.dropna().sort_values("count", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors_list = [PALETTE["prophet"], PALETTE["sarimax"],
                   PALETTE["lstm"], PALETTE["accent"]] * 3
    wedges, texts, autotexts = ax.pie(
        df["count"], labels=df["industry"],
        colors=colors_list[:len(df)],
        autopct="%1.1f%%", startangle=140,
        pctdistance=0.82, wedgeprops={"edgecolor": "white", "linewidth": 1.5}
    )

    for t in autotexts:
        t.set_fontsize(9)

    ax.set_title("Job Postings by Industry")
    _save(fig, os.path.join(out_dir, "16_top_industries.png"))


# ── 17. Skill trends heatmap ───────────────────────────────────────────────────

def plot_skill_trends_heatmap(skill_trends_path, out_dir, top_n=15):
    df = _load(skill_trends_path, index_col=0, parse_dates=True)
    if df.empty:
        return

    # Keep top N skills by total mentions
    top_skills = df.sum().sort_values(ascending=False).head(top_n).index.tolist()
    df = df[top_skills].T  # shape: (skills, months)

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(df.values, aspect="auto",
                   cmap="YlOrRd", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Mention count")

    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index, fontsize=9)
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(
        [str(c)[:7] for c in df.columns],
        rotation=45, ha="right", fontsize=8
    )
    ax.set_title(f"Monthly Skill Demand Heatmap (top {top_n} skills)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Skill")
    _save(fig, os.path.join(out_dir, "17_skill_trends_heatmap.png"))


# ── Master runner ──────────────────────────────────────────────────────────────

def run_all(processed_dir="data/processed",
            predictions_forecasts_dir="predictions/forecasts",
            predictions_validation_dir="predictions/validation",
            out_dir="visualizations"):
    os.makedirs(out_dir, exist_ok=True)

    # --- Load demand series ---
    monthly_series = pd.Series(dtype=float)
    demand_path = os.path.join(processed_dir, "monthly_demand_series.csv")

    try:
        df = pd.read_csv(demand_path, index_col=0, parse_dates=True)
        col = "y" if "y" in df.columns else df.columns[0]
        monthly_series = df[col]
        monthly_series.index = pd.to_datetime(monthly_series.index)
    except FileNotFoundError:
        logger.warning("monthly_demand_series.csv not found.")

    # --- Load processed outputs ---
    def load_dated(path):
        df = _load(path, index_col=0, parse_dates=True)
        if not df.empty:
            df.index = pd.to_datetime(df.index)

        return df

    # Forecasts from predictions/forecasts/
    prophet_df = _load(os.path.join(
        predictions_forecasts_dir, "prophet_forecast.csv"))
    sarimax_df = load_dated(os.path.join(
        predictions_forecasts_dir, "sarimax_forecast.csv"))
    lstm_df = _load(os.path.join(
        predictions_forecasts_dir, "lstm_forecast.csv"))
    scenario_df = load_dated(os.path.join(
        predictions_forecasts_dir, "scenario_forecasts.csv"))

    # Seasonality from data/processed/
    season_df = _load(os.path.join(processed_dir, "seasonality_by_month.csv"))

    # Backtests from predictions/validation/
    prophet_bt = _load(os.path.join(
        predictions_validation_dir, "prophet_backtest.csv"))
    sarimax_bt = _load(os.path.join(
        predictions_validation_dir, "sarimax_backtest.csv"))
    lstm_bt = _load(os.path.join(
        predictions_validation_dir, "lstm_backtest.csv"))

    def load_metrics(path):
        df = _load(path, index_col=0, header=0)
        if df.empty:
            return {}

        return {str(idx): float(row.iloc[0]) for idx, row in df.iterrows()}

    prophet_metrics = load_metrics(os.path.join(
        predictions_validation_dir, "prophet_backtest_metrics.csv"))
    sarimax_metrics = load_metrics(os.path.join(
        predictions_validation_dir, "sarimax_backtest_metrics.csv"))
    lstm_metrics = load_metrics(os.path.join(
        predictions_validation_dir, "lstm_backtest_metrics.csv"))

    # --- 1. Demand overview ---
    if not monthly_series.empty:
        plot_demand_overview(monthly_series, out_dir)
        plot_decomposition(monthly_series, out_dir)

    # --- 3. Seasonality by month ---
    plot_seasonality_by_month(season_df, out_dir)

    # --- 4-6. Individual forecasts ---
    if not monthly_series.empty:
        plot_prophet_forecast(prophet_df, monthly_series, out_dir)
        plot_sarimax_forecast(sarimax_df, monthly_series, out_dir)
        plot_lstm_forecast(lstm_df, monthly_series, out_dir)

        # --- 7. Comparison ---
        plot_forecast_comparison(prophet_df.copy() if not prophet_df.empty else prophet_df,
                                 sarimax_df.copy() if not sarimax_df.empty else sarimax_df,
                                 lstm_df.copy() if not lstm_df.empty else lstm_df,
                                 monthly_series, out_dir)

    # --- 8. Scenarios ---
    if not monthly_series.empty:
        plot_scenarios(scenario_df, monthly_series, out_dir)

    # --- 9-11. Individual backtests ---
    plot_backtest(prophet_bt, prophet_metrics, "Prophet", PALETTE["prophet"],
                  out_dir, "09_prophet_backtest.png")
    plot_backtest(sarimax_bt, sarimax_metrics, "SARIMAX", PALETTE["sarimax"],
                  out_dir, "10_sarimax_backtest.png")
    plot_backtest(lstm_bt, lstm_metrics, "LSTM", PALETTE["lstm"],
                  out_dir, "11_lstm_backtest.png")

    # --- 12. Backtest comparison ---
    plot_backtest_comparison([
        (prophet_bt, "Prophet", PALETTE["prophet"], prophet_metrics),
        (sarimax_bt, "SARIMAX", PALETTE["sarimax"], sarimax_metrics),
        (lstm_bt,    "LSTM",    PALETTE["lstm"],    lstm_metrics),
    ], out_dir)

    # --- 13. Metrics comparison ---
    plot_metrics_comparison({
        "Prophet": prophet_metrics,
        "SARIMAX": sarimax_metrics,
        "LSTM":    lstm_metrics,
    }, out_dir)

    # --- 14-16. NLP results (from data/processed/) ---
    plot_top_skills(os.path.join(processed_dir, "top_skills.csv"), out_dir)
    plot_emerging_roles(os.path.join(
        processed_dir, "emerging_roles.csv"), out_dir)
    plot_top_industries(os.path.join(
        processed_dir, "top_industries.csv"), out_dir)

    # --- 17. Skill heatmap ---
    plot_skill_trends_heatmap(os.path.join(
        processed_dir, "skill_trends_monthly.csv"), out_dir)

    logger.info("All visualizations saved to '%s/'", out_dir)


if __name__ == "__main__":
    PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    run_all(
        processed_dir=os.path.join(PARENT_DIR, "data/processed"),
        predictions_forecasts_dir=os.path.join(
            PARENT_DIR, "predictions/forecasts"),
        predictions_validation_dir=os.path.join(
            PARENT_DIR, "predictions/validation"),
        out_dir=os.path.join(PARENT_DIR, "visualizations"),
    )
