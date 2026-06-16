"""
validation.py
---------------
Validates forecasting model accuracy on historical data via backtesting
(holdout split) using MAE, MAPE, and RMSE.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0

    if not mask.any():
        return np.nan

    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)

    return {"MAE": mae, "RMSE": rmse, "MAPE_%": mape}


def backtest_prophet(merged_df, regressor_cols, holdout_periods=12,
                     seasonality_mode="multiplicative"):
    """
    Train Prophet on all-but-last `holdout_periods` rows, predict the held-out
    period, and compute accuracy metrics against actuals.
    """

    from prophet import Prophet

    if len(merged_df) <= holdout_periods + 12:
        logger.warning(
            "Not enough data for backtest (have %d rows, need > %d).",
            len(merged_df), holdout_periods + 12
        )
        return None, None

    train = merged_df.iloc[:-holdout_periods]
    test = merged_df.iloc[-holdout_periods:]

    train_prophet = train.reset_index().rename(
        columns={train.index.name or "index": "ds"})
    train_prophet = train_prophet.rename(
        columns={train_prophet.columns[0]: "ds"})

    model = Prophet(seasonality_mode=seasonality_mode, yearly_seasonality=True)
    valid_regressors = [
        c for c in regressor_cols if c in train_prophet.columns]

    for col in valid_regressors:
        model.add_regressor(col)

    model.fit(train_prophet)

    test_prophet = test.reset_index().rename(
        columns={test.index.name or "index": "ds"})
    test_prophet = test_prophet.rename(columns={test_prophet.columns[0]: "ds"})

    forecast = model.predict(test_prophet[["ds"] + valid_regressors])

    metrics = compute_metrics(test["y"].values, forecast["yhat"].values)
    logger.info("Prophet backtest metrics: %s", metrics)

    result_df = pd.DataFrame({
        "ds": test_prophet["ds"].values,
        "actual": test["y"].values,
        "predicted": forecast["yhat"].values,
    })

    return metrics, result_df


def backtest_sarimax(merged_df, exog_cols, holdout_periods=12,
                     order=(1, 1, 1), seasonal_order=(0, 1, 1, 12)):
    """Train SARIMAX on train split, forecast holdout, compute metrics."""

    from statsmodels.tsa.statespace.sarimax import SARIMAX

    if len(merged_df) <= holdout_periods + 12:
        logger.warning(
            "Not enough data for backtest (have %d rows, need > %d).",
            len(merged_df), holdout_periods + 12
        )
        return None, None

    train = merged_df.iloc[:-holdout_periods]
    test = merged_df.iloc[-holdout_periods:]

    if exog_cols:
        exog_train_raw = train[exog_cols].astype(float)
        exog_mean = exog_train_raw.mean()
        exog_std = exog_train_raw.std().replace(0, 1)
        exog_train = (exog_train_raw - exog_mean) / exog_std
        exog_test = (test[exog_cols].astype(float) - exog_mean) / exog_std
    else:
        exog_train = None
        exog_test = None

    model = SARIMAX(
        train["y"], exog=exog_train, order=order, seasonal_order=seasonal_order,
        enforce_stationarity=False, enforce_invertibility=False, concentrate_scale=True,
    )
    fitted = model.fit(disp=False, maxiter=200, method="lbfgs")

    forecast = fitted.get_forecast(steps=holdout_periods, exog=exog_test)
    predicted = forecast.predicted_mean.values

    metrics = compute_metrics(test["y"].values, predicted)
    logger.info("SARIMAX backtest metrics: %s", metrics)

    result_df = pd.DataFrame({
        "ds": test.index,
        "actual": test["y"].values,
        "predicted": predicted,
    })

    return metrics, result_df


if __name__ == "__main__":
    import os

    PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PROCESSED = os.path.join(PARENT_DIR, "data", "processed")
    FORECAST_DIR = os.path.join(PARENT_DIR, "predictions", "forecasts")
    VALIDATION_DIR = os.path.join(
        PARENT_DIR, "predictions", "validation")

    os.makedirs(VALIDATION_DIR, exist_ok=True)

    merged = pd.read_csv(
        os.path.join(DATA_PROCESSED, "merged_features.csv"), index_col=0, parse_dates=True)
    regressor_cols = [c for c in merged.columns if c != "y"][:3]

    prophet_metrics, prophet_results = backtest_prophet(
        merged, regressor_cols, holdout_periods=6)
    if prophet_results is not None:
        prophet_results.to_csv(
            os.path.join(VALIDATION_DIR, "prophet_backtest.csv"), index=False)
        pd.Series(prophet_metrics).to_csv(
            os.path.join(VALIDATION_DIR, "prophet_backtest_metrics.csv"))

    sarimax_metrics, sarimax_results = backtest_sarimax(
        merged, regressor_cols, holdout_periods=6)
    if sarimax_results is not None:
        sarimax_results.to_csv(
            os.path.join(VALIDATION_DIR, "sarimax_backtest.csv"), index=False)
        pd.Series(sarimax_metrics).to_csv(
            os.path.join(VALIDATION_DIR, "sarimax_backtest_metrics.csv"))
