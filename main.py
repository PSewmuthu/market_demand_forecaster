"""
main.py
---------
Orchestrates the full Market Demand Trend Analysis pipeline:

1. Data collection: pytrends (search interest), job boards, economic indicators
2. NLP: extract skills/roles/industries from job postings (snapshot-based)
3. Analysis: demand proxy + seasonality decomposition (from Google Trends history)
4. Forecasting: Prophet + SARIMAX + LSTM with external regressors
5. Scenario analysis: best/worst/most-likely
6. Validation: backtest accuracy (MAE, RMSE, MAPE)

Note on data sources:
  Job board APIs (Adzuna/Remotive) return only currently-live postings, so a
  "monthly posting count" derived from them is a snapshot artifact dominated
  by the last 1-3 months, not a real historical time series. Job postings are
  therefore used only for the NLP skill/role/industry extraction (step 2).

  The forecasting target ("y") is instead built from Google Trends search-interest
  data (trends_industries.csv by default), which provides a genuine multi-year
  weekly history. trends_roles.csv and trends_skills.csv are aggregated into
  regressor signals alongside economic indicators.

Run from the project root: python main.py
"""

import os
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(ROOT_DIR, "data", "raw")
DATA_PROCESSED = os.path.join(ROOT_DIR, "data", "processed")
FORECASTS_DIR = os.path.join(ROOT_DIR, "predictions", "forecasts")
VALIDATION_DIR = os.path.join(ROOT_DIR, "predictions", "validation")

os.makedirs(DATA_RAW, exist_ok=True)
os.makedirs(DATA_PROCESSED, exist_ok=True)
os.makedirs(FORECASTS_DIR, exist_ok=True)
os.makedirs(VALIDATION_DIR, exist_ok=True)


def step_collect_data(run_pytrends=True, run_jobs=True, run_econ=True):
    logger.info("=== STEP 1: Data Collection ===")

    if run_pytrends:
        from data_collection.pytrends_collector import collect_all_trends, save_trends

        trends = collect_all_trends()
        save_trends(trends, output_dir=DATA_RAW)

    if run_jobs:
        from data_collection.job_board_collector import collect_all_jobs

        jobs_df = collect_all_jobs()
        if not jobs_df.empty:
            jobs_df.to_csv(os.path.join(
                DATA_RAW, "job_postings.csv"), index=False)
            logger.info("Saved %d job postings.", len(jobs_df))
        else:
            logger.warning("No job postings collected.")

    if run_econ:
        from data_collection.economic_collector import collect_economic_indicators

        econ_df = collect_economic_indicators()
        if not econ_df.empty:
            econ_df.to_csv(os.path.join(DATA_RAW, "economic_indicators.csv"))
            logger.info("Saved economic indicators.")
        else:
            logger.warning("No economic data collected (check FRED_API_KEY).")


def step_nlp_processing():
    logger.info("=== STEP 2: NLP Skill/Role/Industry Extraction ===")

    from nlp.skill_extraction import (
        add_skill_column, top_skills, emerging_roles, top_industries, skill_trend_by_month
    )

    jobs_path = os.path.join(DATA_RAW, "job_postings.csv")
    if not os.path.exists(jobs_path):
        logger.warning("No job postings file found, skipping NLP step.")
        return None

    jobs_df = pd.read_csv(jobs_path)
    jobs_df = add_skill_column(jobs_df)

    logger.info("Top skills:\n%s", top_skills(jobs_df))
    logger.info("Emerging roles:\n%s", emerging_roles(jobs_df))
    logger.info("Top industries:\n%s", top_industries(jobs_df))

    jobs_df.to_csv(os.path.join(
        DATA_PROCESSED, "jobs_with_skills.csv"), index=False)
    skill_trends = skill_trend_by_month(jobs_df)
    skill_trends.to_csv(os.path.join(
        DATA_PROCESSED, "skill_trends_monthly.csv"))

    top_skills(jobs_df).to_csv(os.path.join(DATA_PROCESSED, "top_skills.csv"))
    emerging_roles(jobs_df).to_csv(os.path.join(
        DATA_PROCESSED, "emerging_roles.csv"))
    top_industries(jobs_df).to_csv(os.path.join(
        DATA_PROCESSED, "top_industries.csv"))

    return jobs_df


def step_seasonality_analysis(jobs_df, target_category="industries"):
    logger.info("=== STEP 3: Seasonality Analysis ===")

    from analysis.demand_target import build_monthly_demand_series
    from analysis.revenue_seasonality import decompose_series, plot_decomposition, seasonality_summary

    # Build the primary demand proxy from Google Trends data.
    # Job board snapshots only capture currently-live postings (dominated by
    # the last 1-3 months), so they are NOT used as the forecasting target --
    # they remain useful for the NLP/skills analysis (step 2) only.
    trends_paths = {
        "roles": f"{DATA_RAW}/trends_roles.csv",
        "skills": f"{DATA_RAW}/trends_skills.csv",
        "industries": f"{DATA_RAW}/trends_industries.csv",
    }

    try:
        monthly_series = build_monthly_demand_series(
            {target_category: trends_paths[target_category]}, agg="mean", monthly_agg="mean"
        )
    except FileNotFoundError as exc:
        logger.warning(
            "Could not build demand series from trends data: %s", exc)
        return None

    monthly_series.to_csv(os.path.join(
        DATA_PROCESSED, "monthly_demand_series.csv"))

    if len(monthly_series) >= 24:
        result = decompose_series(monthly_series, period=12)
        plot_decomposition(
            result, save_path=os.path.join(DATA_PROCESSED, "seasonality_decomposition.png"))
        summary = seasonality_summary(result)
        summary.to_csv(os.path.join(
            DATA_PROCESSED, "seasonality_by_month.csv"))
        logger.info("Seasonality summary:\n%s", summary)
    else:
        logger.warning(
            "Not enough monthly data (%d) for seasonal decomposition.", len(monthly_series))

    return monthly_series


def step_forecasting(monthly_series):
    logger.info("=== STEP 4: Forecasting ===")

    from analysis.forecasting_models import merge_datasets, train_prophet_model, forecast_prophet, train_sarimax_model, forecast_sarimax

    if monthly_series is None or len(monthly_series) < 24:
        logger.warning("Insufficient data for forecasting. Need >= 24 months.")
        return None, None, None, None

    # Use the trend categories NOT used as the forecast target as regressors
    # (the "industries" trend series is the target itself in step 3, so
    # roles/skills trends are used here to avoid leakage).
    try:
        roles = pd.read_csv(os.path.join(DATA_RAW, "trends_roles.csv"),
                            index_col=0, parse_dates=True)
        # Collapse to a single "roles demand" signal to keep regressor count small
        roles_signal = roles.mean(axis=1).rename(
            "roles_trend").resample("ME").mean()
    except FileNotFoundError:
        roles_signal = pd.Series(dtype=float, name="roles_trend")

    try:
        skills = pd.read_csv(
            os.path.join(DATA_RAW, "trends_skills.csv"), index_col=0, parse_dates=True)
        skills_signal = skills.mean(axis=1).rename(
            "skills_trend").resample("ME").mean()
    except FileNotFoundError:
        skills_signal = pd.Series(dtype=float, name="skills_trend")

    trends = pd.concat([roles_signal, skills_signal], axis=1)
    trends = trends.dropna(how="all")

    try:
        econ = pd.read_csv(
            os.path.join(DATA_RAW, "economic_indicators.csv"), index_col=0, parse_dates=True)
        if not isinstance(econ.index, pd.DatetimeIndex) or econ.index.isna().all():
            raise ValueError(
                "economic_indicators.csv has no usable date index")
    except (FileNotFoundError, ValueError) as exc:
        logger.warning(
            "Economic indicators unavailable or missing date index: %s", exc)
        econ = pd.DataFrame()

    merged = merge_datasets(monthly_series, trends, econ)
    merged.to_csv(os.path.join(DATA_PROCESSED, "merged_features.csv"))

    regressor_candidates = [c for c in merged.columns if c != "y"][:5]

    model, prophet_df, used_regressors = train_prophet_model(
        merged, regressor_cols=regressor_candidates)
    forecast = forecast_prophet(model, prophet_df, used_regressors, periods=12)
    forecast.to_csv(os.path.join(
        FORECASTS_DIR, "prophet_forecast.csv"), index=False)
    logger.info("Prophet forecast saved.")

    sarimax_fitted = train_sarimax_model(merged, exog_cols=used_regressors)
    sarimax_mean, sarimax_ci = forecast_sarimax(
        sarimax_fitted, merged, exog_cols=used_regressors)
    sarimax_result = pd.concat(
        [sarimax_mean.rename("forecast"), sarimax_ci], axis=1)
    sarimax_result.to_csv(os.path.join(FORECASTS_DIR, "sarimax_forecast.csv"))
    logger.info("SARIMAX forecast saved.")

    # --- LSTM model ---
    try:
        from analysis.lstm_model import train_lstm_model, forecast_lstm

        lookback = min(6, max(2, len(merged) // 4))
        test_size = min(6, max(1, len(merged) // 6))

        lstm_trained = train_lstm_model(
            merged, lookback=lookback, test_size=test_size,
            feature_cols=used_regressors,
        )
        lstm_forecast = forecast_lstm(lstm_trained, merged, periods=12)
        lstm_forecast.to_csv(
            os.path.join(FORECASTS_DIR, "lstm_forecast.csv"), index=False)
        logger.info("LSTM forecast saved.")
    except ImportError:
        logger.warning(
            "TensorFlow not installed. Skipping LSTM model (pip install tensorflow).")
    except ValueError as exc:
        logger.warning("LSTM training skipped: %s", exc)

    return merged, model, prophet_df, used_regressors


def step_scenario_analysis(model, prophet_df, used_regressors):
    logger.info("=== STEP 5: Scenario Analysis ===")

    from analysis.scenario_analysis import run_prophet_scenarios, combine_scenarios_to_df

    if model is None:
        logger.warning(
            "No trained model available, skipping scenario analysis.")
        return

    scenario_results = run_prophet_scenarios(
        model, prophet_df, used_regressors, periods=12)
    combined = combine_scenarios_to_df(scenario_results)
    combined.to_csv(os.path.join(FORECASTS_DIR, "scenario_forecasts.csv"))
    logger.info("Scenario forecasts saved.")


def step_validation(merged):
    logger.info("=== STEP 6: Validation / Backtesting ===")

    from analysis.validation import backtest_prophet, backtest_sarimax

    if merged is None:
        logger.warning("No merged dataset available, skipping validation.")
        return

    regressor_cols = [c for c in merged.columns if c != "y"][:5]

    prophet_metrics, prophet_results = backtest_prophet(
        merged, regressor_cols, holdout_periods=6)
    if prophet_results is not None:
        prophet_results.to_csv(
            os.path.join(VALIDATION_DIR, "prophet_backtest.csv"), index=False)
        pd.Series(prophet_metrics).to_csv(
            os.path.join(VALIDATION_DIR, "prophet_backtest_metrics.csv"))
        logger.info("Prophet backtest metrics: %s", prophet_metrics)

    sarimax_metrics, sarimax_results = backtest_sarimax(
        merged, regressor_cols, holdout_periods=6)
    if sarimax_results is not None:
        sarimax_results.to_csv(
            os.path.join(VALIDATION_DIR, "sarimax_backtest.csv"), index=False)
        pd.Series(sarimax_metrics).to_csv(
            os.path.join(VALIDATION_DIR, "sarimax_backtest_metrics.csv"))
        logger.info("SARIMAX backtest metrics: %s", sarimax_metrics)

    # --- LSTM backtest ---
    try:
        from analysis.lstm_model import train_lstm_model, evaluate_lstm

        lookback = min(6, max(2, len(merged) // 4))
        test_size = min(6, max(1, len(merged) // 6))

        lstm_trained = train_lstm_model(
            merged, lookback=lookback, test_size=test_size,
            feature_cols=regressor_cols,
        )
        lstm_metrics, lstm_results = evaluate_lstm(lstm_trained, merged)
        lstm_results.to_csv(os.path.join(
            VALIDATION_DIR, "lstm_backtest.csv"), index=False)
        pd.Series(lstm_metrics).to_csv(
            os.path.join(VALIDATION_DIR, "lstm_backtest_metrics.csv"))
        logger.info("LSTM backtest metrics: %s", lstm_metrics)
    except ImportError:
        logger.warning(
            "TensorFlow not installed. Skipping LSTM backtest (pip install tensorflow).")
    except ValueError as exc:
        logger.warning("LSTM backtest skipped: %s", exc)


def main():
    # Toggle steps as needed; data collection can be slow/rate-limited
    # step_collect_data(run_pytrends=True, run_jobs=True, run_econ=True)

    jobs_df = step_nlp_processing()
    monthly_series = step_seasonality_analysis(jobs_df)
    merged, model, prophet_df, used_regressors = step_forecasting(
        monthly_series)
    step_scenario_analysis(model, prophet_df, used_regressors)
    step_validation(merged)

    logger.info("\n=== Pipeline complete. Outputs in predictions/ ===\n")


if __name__ == "__main__":
    main()
