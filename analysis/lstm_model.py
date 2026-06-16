"""
lstm_model.py
---------------
LSTM-based time series forecasting for market demand, supporting
external regressors (search trends, economic indicators) as additional
input features alongside the target series.

Why LSTM here:
  - Captures nonlinear relationships between demand and external factors
    that Prophet/SARIMAX (largely additive/linear) can miss.
  - Useful when SARIMAX fails to converge or Prophet's trend/seasonality
    assumptions don't fit the data well.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
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


def create_sequences(data, lookback):
    """
    Convert a 2D array (rows=time, cols=features) into overlapping sequences.

    Returns:
        X: shape (n_samples, lookback, n_features)
        y: shape (n_samples,) -- the target value (column 0) at the step after each sequence
    """

    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i + lookback, :])
        y.append(data[i + lookback, 0])  # target = column 0 ("y")

    return np.array(X), np.array(y)


def prepare_lstm_data(merged_df, lookback=12, feature_cols=None):
    """
    Scale features and the target, then build sequences for LSTM training.

    merged_df: DataFrame with column 'y' (target) and optional regressor columns,
               DatetimeIndex, monthly frequency.
    feature_cols: list of regressor column names to include (target 'y' is always included
                   as column 0). If None, all non-'y' columns are used.

    Returns:
        X, y_arr, feature_scaler, target_scaler, feature_cols, data_scaled
    """

    feature_cols = feature_cols if feature_cols is not None else [
        c for c in merged_df.columns if c != "y"]

    # Defend against all-NaN columns
    valid_feature_cols = []
    for col in feature_cols:
        if merged_df[col].isna().all():
            logger.warning(
                "Dropping regressor '%s' from LSTM features: column is entirely NaN "
                "(likely a stale/incomplete merged_features.csv -- consider "
                "re-running the pipeline to regenerate it).",
                col,
            )
        else:
            valid_feature_cols.append(col)
    feature_cols = valid_feature_cols

    ordered_cols = ["y"] + feature_cols

    data = merged_df[ordered_cols].astype(float).values

    # Fill any remaining (partial) NaNs via forward/backward fill so a few
    # missing values don't propagate NaN through the whole scaled array.
    if np.isnan(data).any():
        data = (
            pd.DataFrame(data, columns=ordered_cols, index=merged_df.index)
            .ffill().bfill().values
        )

    target_scaler = MinMaxScaler()
    feature_scaler = MinMaxScaler()

    target_scaled = target_scaler.fit_transform(data[:, [0]])
    if feature_cols:
        features_scaled = feature_scaler.fit_transform(data[:, 1:])
        data_scaled = np.hstack([target_scaled, features_scaled])
    else:
        data_scaled = target_scaled
        feature_scaler = None

    X, y_arr = create_sequences(data_scaled, lookback)
    return X, y_arr, feature_scaler, target_scaler, feature_cols, data_scaled


def build_lstm_model(n_features, lookback, units=32, dropout=0.2, learning_rate=0.001,
                     l2_reg=1e-4):
    """Build and compile a (size-appropriate) LSTM model.

    For small datasets, a single small LSTM layer generalizes far better than
    a deep stack -- an over-parameterized model on ~20-30 training sequences
    will collapse to predicting the training-set mean for every input.
    """

    from tensorflow import keras
    from tensorflow.keras import layers, regularizers

    reg = regularizers.l2(l2_reg) if l2_reg else None

    if units <= 16:
        # Tiny dataset: single small layer, heavier regularization
        model = keras.Sequential([
            layers.Input(shape=(lookback, n_features)),
            layers.LSTM(units, kernel_regularizer=reg),
            layers.Dropout(dropout),
            layers.Dense(1),
        ])
    else:
        model = keras.Sequential([
            layers.Input(shape=(lookback, n_features)),
            layers.LSTM(units, return_sequences=True, kernel_regularizer=reg),
            layers.Dropout(dropout),
            layers.LSTM(max(8, units // 2), kernel_regularizer=reg),
            layers.Dropout(dropout),
            layers.Dense(16, activation="relu"),
            layers.Dense(1),
        ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def _auto_hyperparams(n_train_sequences):
    """
    Pick model size / batch size / patience based on how many training
    sequences are available. Small datasets get a small model -- otherwise
    the LSTM overfits to a constant (the training-set mean) almost immediately.
    """

    if n_train_sequences < 15:
        return {"units": 8, "batch_size": 1, "patience": 30, "epochs": 300}
    elif n_train_sequences < 40:
        return {"units": 16, "batch_size": 2, "patience": 25, "epochs": 250}
    elif n_train_sequences < 100:
        return {"units": 32, "batch_size": 4, "patience": 20, "epochs": 200}
    else:
        return {"units": 64, "batch_size": 8, "patience": 20, "epochs": 200}


def _is_collapsed(preds, threshold=1e-3):
    """Check if predictions are (near) constant -- a sign the model
    learned a trivial constant-output solution rather than real dynamics."""

    preds = np.asarray(preds).flatten()
    return (preds.max() - preds.min()) < threshold


def train_lstm_model(merged_df, lookback=12, test_size=6, units=None, dropout=0.2,
                     epochs=None, batch_size=None, learning_rate=0.001, feature_cols=None,
                     patience=None, verbose=0, n_restarts=5, random_seed=None):
    """
    Train an LSTM model on the merged dataset.

    Splits the last `test_size` time steps off as a holdout test set
    (the sequences whose targets fall within that window).

    If units/batch_size/patience/epochs are not specified, they are chosen
    automatically based on the number of training sequences -- small datasets
    get a small model, since an over-parameterized LSTM on ~10-20 sequences
    will collapse to predicting a constant (the training-set mean).

    Returns a dict with model, scalers, history, and train/test arrays for evaluation.
    """

    from tensorflow import keras

    X, y_arr, feature_scaler, target_scaler, feature_cols, data_scaled = prepare_lstm_data(
        merged_df, lookback=lookback, feature_cols=feature_cols
    )

    if len(X) <= test_size:
        raise ValueError(
            f"Not enough sequences ({len(X)}) for test_size={test_size}. "
            f"Need more historical data or a smaller lookback/test_size."
        )

    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y_arr[:-test_size], y_arr[-test_size:]

    auto = _auto_hyperparams(len(X_train))
    units = units if units is not None else auto["units"]
    batch_size = batch_size if batch_size is not None else auto["batch_size"]
    patience = patience if patience is not None else auto["patience"]
    epochs = epochs if epochs is not None else auto["epochs"]

    if len(X_train) < 15:
        logger.warning(
            "Only %d training sequences available. LSTM forecasts will be unreliable "
            "with this little data -- treat results as illustrative only. "
            "Prophet/SARIMAX are likely more trustworthy until more history is collected.",
            len(X_train),
        )

    n_features = X.shape[2]

    best_model = None
    best_history = None
    best_val_loss = np.inf
    best_is_collapsed = True

    for attempt in range(n_restarts):
        if random_seed is not None:
            keras.utils.set_random_seed(random_seed + attempt)

        model = build_lstm_model(n_features, lookback, units=units, dropout=dropout,
                                 learning_rate=learning_rate)

        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        )

        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=verbose,
        )

        train_preds = model.predict(X_train, verbose=0).flatten()
        collapsed = _is_collapsed(train_preds)
        val_loss = min(history.history.get("val_loss", [np.inf]))

        logger.info(
            "LSTM attempt %d/%d: val_loss=%.5f, collapsed=%s",
            attempt + 1, n_restarts, val_loss, collapsed,
        )

        # Prefer non-collapsed models; among those, lowest val_loss wins.
        # If all attempts collapse, keep the best val_loss anyway as a fallback.
        is_better = (
            (best_is_collapsed and not collapsed)
            or (collapsed == best_is_collapsed and val_loss < best_val_loss)
        )
        if is_better:
            best_model = model
            best_history = history
            best_val_loss = val_loss
            best_is_collapsed = collapsed

        if not collapsed:
            break  # good enough, stop restarting

    if best_is_collapsed:
        logger.warning(
            "All %d LSTM training attempts collapsed to a near-constant output. "
            "This usually means there isn't enough data/signal for the LSTM to learn "
            "temporal dynamics. Forecast will be unreliable -- rely on Prophet/SARIMAX, "
            "or collect more historical data before trusting LSTM results.",
            n_restarts,
        )

    model = best_model
    history = best_history

    return {
        "model": model,
        "history": history,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "feature_cols": feature_cols,
        "lookback": lookback,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "data_scaled": data_scaled,
        "is_collapsed": best_is_collapsed,
    }


def evaluate_lstm(trained, merged_df):
    """
    Evaluate the trained LSTM on its holdout test set.
    Returns metrics dict and a DataFrame of actual vs predicted values
    (inverse-scaled to original units).
    """

    model = trained["model"]
    target_scaler = trained["target_scaler"]
    X_test, y_test = trained["X_test"], trained["y_test"]
    lookback = trained["lookback"]

    y_pred_scaled = model.predict(X_test, verbose=0).flatten()

    y_test_actual = target_scaler.inverse_transform(
        y_test.reshape(-1, 1)).flatten()
    y_pred_actual = target_scaler.inverse_transform(
        y_pred_scaled.reshape(-1, 1)).flatten()

    mae = mean_absolute_error(y_test_actual, y_pred_actual)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    mape = mean_absolute_percentage_error(y_test_actual, y_pred_actual)
    metrics = {"MAE": mae, "RMSE": rmse, "MAPE_%": mape}

    test_size = len(y_test)
    dates = merged_df.index[-test_size:]

    result_df = pd.DataFrame({
        "ds": dates,
        "actual": y_test_actual,
        "predicted": y_pred_actual,
    })

    logger.info("LSTM backtest metrics: %s", metrics)
    return metrics, result_df


def forecast_lstm(trained, merged_df, periods=12):
    """
    Generate a multi-step-ahead forecast by recursively feeding predictions
    back into the model.

    Future regressor values are carried forward using the last observed
    (scaled) values -- same naive assumption used elsewhere in this project.

    Returns a DataFrame with columns ['ds', 'yhat'].
    """

    model = trained["model"]
    target_scaler = trained["target_scaler"]
    feature_scaler = trained["feature_scaler"]
    feature_cols = trained["feature_cols"]
    lookback = trained["lookback"]
    data_scaled = trained["data_scaled"]

    if trained.get("is_collapsed"):
        logger.warning(
            "Generating LSTM forecast from a collapsed model -- output is likely "
            "a near-constant value and should not be used for decision-making. "
            "Prefer the Prophet/SARIMAX forecasts instead."
        )

    n_features = data_scaled.shape[1]
    window = data_scaled[-lookback:, :].copy()

    last_future_features = window[-1, 1:].copy() if feature_cols else None

    preds_scaled = []
    for _ in range(periods):
        x_input = window.reshape(1, lookback, n_features)
        next_scaled = model.predict(x_input, verbose=0)[0, 0]
        preds_scaled.append(next_scaled)

        if feature_cols:
            next_row = np.concatenate([[next_scaled], last_future_features])
        else:
            next_row = np.array([next_scaled])

        window = np.vstack([window[1:], next_row])

    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    preds_actual = target_scaler.inverse_transform(preds_scaled).flatten()

    last_date = merged_df.index[-1]
    future_dates = pd.date_range(
        start=last_date, periods=periods + 1, freq="ME")[1:]

    return pd.DataFrame({"ds": future_dates, "yhat": preds_actual})


if __name__ == "__main__":
    import os

    PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PROCESSED = os.path.join(PARENT_DIR, "data", "processed")
    FORECAST_DIR = os.path.join(PARENT_DIR, "predictions", "forecasts")
    VALIDATION_DIR = os.path.join(PARENT_DIR, "predictions", "validation")

    merged = pd.read_csv(
        os.path.join(DATA_PROCESSED, "merged_features.csv"), index_col=0, parse_dates=True)

    if len(merged) < 24:
        logger.warning(
            "Not enough data (%d rows) for a meaningful LSTM. Need >= 24 months.", len(merged))
    else:
        lookback = min(6, max(2, len(merged) // 4))
        test_size = min(6, max(1, len(merged) // 6))

        trained = train_lstm_model(
            merged, lookback=lookback, test_size=test_size)

        metrics, backtest_df = evaluate_lstm(trained, merged)
        backtest_df.to_csv(os.path.join(
            VALIDATION_DIR, "lstm_backtest.csv"), index=False)
        pd.Series(metrics).to_csv(
            os.path.join(VALIDATION_DIR, "lstm_backtest_metrics.csv"))

        forecast_df = forecast_lstm(trained, merged, periods=12)
        forecast_df.to_csv(os.path.join(
            FORECAST_DIR, "lstm_forecast.csv"), index=False)

        logger.info("Saved LSTM forecast and backtest results.")
