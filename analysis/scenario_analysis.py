"""
scenario_analysis.py
-----------------------
Generates best-case, worst-case, and most-likely scenario forecasts
by adjusting external regressors (e.g. economic indicators, trend signals).
"""

import logging
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# Default multiplicative adjustments applied to each regressor per scenario.
# Tune these based on domain knowledge (e.g. unemployment +20% in worst case typically suppresses hiring demand).
DEFAULT_SCENARIOS = {
    "best_case": 0.85,
    "most_likely": 1.0,
    "worst_case": 1.20,
}


def build_scenario_future(prophet_df, future_template, regressor_cols, factor):
    """
    Apply a multiplicative factor to regressor columns in the future dataframe
    for rows beyond the historical data range.
    """

    future = future_template.copy()
    cutoff = prophet_df["ds"].max()

    for col in regressor_cols:
        last_value = prophet_df[col].iloc[-1]
        mask = future["ds"] > cutoff
        future.loc[mask, col] = last_value * factor
        future.loc[~mask, col] = prophet_df[col].reindex(
            future.index).fillna(last_value)

    return future


def run_prophet_scenarios(model, prophet_df, regressor_cols, periods=12, freq="ME",
                          scenarios=None):
    """
    Run scenario forecasts using a fitted Prophet model.

    Returns dict: scenario_name -> forecast DataFrame (ds, yhat, yhat_lower, yhat_upper)
    """

    scenarios = scenarios or DEFAULT_SCENARIOS
    base_future = model.make_future_dataframe(periods=periods, freq=freq)

    results = {}
    for name, factor in scenarios.items():
        future = build_scenario_future(
            prophet_df, base_future, regressor_cols, factor)
        forecast = model.predict(future)
        results[name] = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
        logger.info("Scenario '%s' (factor=%.2f): final forecast value = %.2f",
                    name, factor, forecast["yhat"].iloc[-1])

    return results


def run_sarimax_scenarios(fitted_model, df, exog_cols, periods=12, scenarios=None):
    """
    Run scenario forecasts using a fitted SARIMAX model by scaling exogenous inputs.
    """

    scenarios = scenarios or DEFAULT_SCENARIOS
    results = {}

    if not exog_cols:
        logger.warning(
            "No exogenous columns provided; scenarios will be identical.")

    last_row = df[exog_cols].iloc[-1] if exog_cols else None

    for name, factor in scenarios.items():
        if exog_cols:
            future_exog = pd.DataFrame(
                [(last_row.values * factor)] * periods, columns=exog_cols
            )
        else:
            future_exog = None

        forecast = fitted_model.get_forecast(steps=periods, exog=future_exog)
        mean = forecast.predicted_mean
        results[name] = mean
        logger.info("Scenario '%s' (factor=%.2f): final forecast value = %.2f",
                    name, factor, mean.iloc[-1])

    return results


def combine_scenarios_to_df(scenario_results, value_col="yhat", date_col="ds"):
    """Combine multiple scenario forecasts into a single wide DataFrame for plotting."""

    combined = None

    for name, df in scenario_results.items():
        series = df.set_index(date_col)[value_col].rename(
            name) if date_col in df.columns else df.rename(name)
        if combined is None:
            combined = series.to_frame()
        else:
            combined = combined.join(series, how="outer")

    return combined


if __name__ == "__main__":
    import os

    PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_RAW = os.path.join(PARENT_DIR, "data", "raw")
    DATA_PROCESSED = os.path.join(PARENT_DIR, "data", "processed")
    FORECAST_DIR = os.path.join(PARENT_DIR, "data", "processed", "forecasts")

    os.makedirs(FORECAST_DIR, exist_ok=True)

    # Expects model/prophet_df/regressors saved from forecasting_models.py run
    # Re-running training here for a standalone demo:
    from forecasting_models import merge_datasets, train_prophet_model

    demand = pd.read_csv(os.path.join(DATA_PROCESSED, "monthly_demand_series.csv"),
                         index_col=0, parse_dates=True)["job_count"]
    try:
        trends = pd.read_csv(os.path.join(DATA_RAW, "trends_roles.csv"),
                             index_col=0, parse_dates=True)
    except FileNotFoundError:
        trends = pd.DataFrame()
    try:
        econ = pd.read_csv(os.path.join(DATA_RAW, "economic_indicators.csv"),
                           index_col=0, parse_dates=True)
    except FileNotFoundError:
        econ = pd.DataFrame()

    merged = merge_datasets(demand, trends, econ)
    regressor_candidates = [c for c in merged.columns if c != "y"][:3]

    model, prophet_df, used_regressors = train_prophet_model(
        merged, regressor_cols=regressor_candidates)
    scenario_results = run_prophet_scenarios(
        model, prophet_df, used_regressors, periods=12)

    combined = combine_scenarios_to_df(scenario_results)
    combined.to_csv(os.path.join(FORECAST_DIR, "scenario_forecasts.csv"))
    logger.info(
        "Saved scenario forecasts -> data/processed/forecasts/scenario_forecasts.csv")
