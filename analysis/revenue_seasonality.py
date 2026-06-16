"""
revenue_seasonality.py
------------------------
Analyzes historical demand (job posting volume / "revenue proxy") patterns
and seasonality using STL/classical decomposition.
"""

import logging
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose, STL

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def decompose_series(series, period=12, model="additive", method="classical"):
    """
    Decompose a time series into trend, seasonal, and residual components.

    method: 'classical' uses statsmodels seasonal_decompose,
            'stl' uses STL (more robust to outliers).
    """

    series = series.asfreq("ME").interpolate()

    if method == "stl":
        stl = STL(series, period=period, robust=True)
        result = stl.fit()

        return result

    result = seasonal_decompose(series, model=model, period=period)

    return result


def plot_decomposition(result, title="Demand Decomposition", save_path=None):
    fig = result.plot()
    fig.suptitle(title)
    fig.set_size_inches(10, 8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
        logger.info("Saved decomposition plot -> %s", save_path)

    plt.close(fig)


def seasonality_summary(result):
    """Return month-of-year average seasonal effect."""

    seasonal = result.seasonal
    df = seasonal.to_frame("seasonal")
    df["month"] = df.index.month

    return df.groupby("month")["seasonal"].mean().sort_values(ascending=False)


if __name__ == "__main__":
    import os

    PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PROCESSED = os.path.join(PARENT_DIR, "data", "processed")

    monthly_series = pd.read_csv(
        os.path.join(DATA_PROCESSED, "monthly_demand_series.csv"), index_col=0, parse_dates=True
    )["y"]

    if len(monthly_series) >= 24:
        result = decompose_series(
            monthly_series, period=12, method="classical")
        plot_decomposition(result, save_path=os.path.join(
            DATA_PROCESSED, "seasonality_decomposition.png"))

        summary = seasonality_summary(result)
        logger.info("Seasonal effect by month:\n%s", summary)
        summary.to_csv(os.path.join(
            DATA_PROCESSED, "seasonality_by_month.csv"))
    else:
        logger.warning(
            "Not enough data points (%d) for seasonal decomposition (need >= 24 months).",
            len(monthly_series),
        )

    logger.info("Saved monthly demand series.")
