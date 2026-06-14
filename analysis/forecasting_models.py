"""
forecasting_models.py
-----------------------
Builds time series forecasting models for market demand using:
  - Prophet (with external regressors)
  - SARIMAX (classical, with exogenous variables)

Both models support incorporating external factors (economic indicators,
Google Trends signals) as regressors/exogenous inputs.
"""

import logging
import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def merge_datasets(demand_series, trends_df=None, econ_df=None):
    """
    Merge demand series with external regressor dataframes on monthly index.

    demand_series: pd.Series, monthly job_count
    trends_df: pd.DataFrame, Google Trends data (any frequency, will be resampled monthly)
    econ_df: pd.DataFrame, economic indicators (monthly)
    """

    df = demand_series.to_frame("y")
    df.index = pd.to_datetime(df.index)

    if trends_df is not None and not trends_df.empty:
        trends_monthly = trends_df.copy()
        trends_monthly.index = pd.to_datetime(
            trends_monthly.index, errors="coerce")
        trends_monthly = trends_monthly[trends_monthly.index.notna()]
        trends_monthly = trends_monthly[~trends_monthly.index.duplicated(
            keep="last")]
        trends_monthly = trends_monthly.sort_index()

        if not trends_monthly.index.is_monotonic_increasing:
            trends_monthly = trends_monthly.iloc[trends_monthly.index.argsort(
            )]

        trends_monthly = trends_monthly.resample("ME").mean()
        df = df.join(trends_monthly, how="left")

    if econ_df is not None and not econ_df.empty:
        econ_monthly = econ_df.copy()
        econ_monthly.index = pd.to_datetime(
            econ_monthly.index, errors="coerce")
        econ_monthly = econ_monthly[econ_monthly.index.notna()]
        econ_monthly = econ_monthly[~econ_monthly.index.duplicated(
            keep="last")]
        econ_monthly = econ_monthly.sort_index()

        if not econ_monthly.index.is_monotonic_increasing:
            econ_monthly = econ_monthly.iloc[econ_monthly.index.argsort()]

        econ_monthly = econ_monthly.resample("ME").ffill()
        df = df.join(econ_monthly, how="left")

    df = df.interpolate().ffill().bfill()
    return df


def train_prophet_model(df, regressor_cols=None, seasonality_mode="multiplicative"):
    """
    Train a Prophet model with optional external regressors.

    df: DataFrame with DateTimeIndex and column 'y' (target), plus regressor columns.
    regressor_cols: list of column names to add as Prophet regressors.
    """

    regressor_cols = regressor_cols or []

    prophet_df = df.reset_index().rename(
        columns={df.index.name or "index": "ds"})
    prophet_df = prophet_df.rename(columns={prophet_df.columns[0]: "ds"})

    model = Prophet(seasonality_mode=seasonality_mode, yearly_seasonality=True)

    for col in regressor_cols:
        if col in prophet_df.columns:
            model.add_regressor(col)
        else:
            logger.warning(
                "Regressor column '%s' not found in dataframe, skipping.", col)
            regressor_cols.remove(col)

    model.fit(prophet_df)
    return model, prophet_df, regressor_cols


def forecast_prophet(model, prophet_df, regressor_cols, periods=12, freq="ME"):
    """
    Generate forecast. Future regressor values are carried forward using
    the last observed value (naive assumption -- replace with real forecasts
    of regressors if available).
    """

    future = model.make_future_dataframe(periods=periods, freq=freq)

    for col in regressor_cols:
        last_value = prophet_df[col].iloc[-1]
        future[col] = prophet_df[col].reindex(future.index).fillna(last_value)
        # Ensure future rows beyond historical data get the last known value
        future.loc[future["ds"] > prophet_df["ds"].max(), col] = last_value

    forecast = model.predict(future)
    return forecast


def train_sarimax_model(df, exog_cols=None, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    """
    Train a SARIMAX model with optional exogenous regressors.
    """

    exog_cols = exog_cols or []
    y = df["y"]
    exog = df[exog_cols] if exog_cols else None

    model = SARIMAX(
        y,
        exog=exog,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False, maxiter=200, method='powell')

    return fitted


def forecast_sarimax(fitted_model, df, exog_cols=None, periods=12):
    """Forecast future periods, carrying forward last exogenous values."""

    exog_cols = exog_cols or []

    if exog_cols:
        last_row = df[exog_cols].iloc[-1]
        future_exog = pd.DataFrame(
            [last_row.values] * periods, columns=exog_cols
        )
    else:
        future_exog = None

    forecast = fitted_model.get_forecast(steps=periods, exog=future_exog)
    mean = forecast.predicted_mean
    ci = forecast.conf_int()

    return mean, ci


if __name__ == "__main__":
    import os

    PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_RAW = os.path.join(PARENT_DIR, "data", "raw")
    DATA_PROCESSED = os.path.join(PARENT_DIR, "data", "processed")
    FORECAST_DIR = os.path.join(PARENT_DIR, "data", "processed", "forecasts")

    os.makedirs(FORECAST_DIR, exist_ok=True)

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
    merged.to_csv(os.path.join(FORECAST_DIR, "merged_features.csv"))

    regressor_candidates = [
        c for c in merged.columns if c != "y"][:3]  # limit for demo

    model, prophet_df, used_regressors = train_prophet_model(
        merged, regressor_cols=regressor_candidates)
    forecast = forecast_prophet(model, prophet_df, used_regressors, periods=12)
    forecast.to_csv(os.path.join(
        FORECAST_DIR, "prophet_forecast.csv"), index=False)
    logger.info("Saved Prophet forecast -> data/processed/prophet_forecast.csv")

    sarimax_fitted = train_sarimax_model(merged, exog_cols=used_regressors)
    sarimax_mean, sarimax_ci = forecast_sarimax(
        sarimax_fitted, merged, exog_cols=used_regressors)
    sarimax_result = pd.concat(
        [sarimax_mean.rename("forecast"), sarimax_ci], axis=1)
    sarimax_result.to_csv(os.path.join(
        FORECAST_DIR, "sarimax_forecast.csv"), index=False)
    logger.info("Saved SARIMAX forecast -> data/processed/sarimax_forecast.csv")
