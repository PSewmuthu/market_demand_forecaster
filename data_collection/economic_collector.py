"""
economic_collector.py
-----------------------
Collects external economic indicators from FRED (Federal Reserve Economic Data)
to use as exogenous regressors in forecasting models.

Get a free API key at: https://fred.stlouisfed.org/docs/api/api_key.html
"""

import os
import logging
import pandas as pd
from fredapi import Fred

try:
    from pathlib import Path
    from dotenv import load_dotenv

    # .parent goes up one level to the root
    dotenv_path = Path(__file__).resolve().parent.parent / '.env'
    # Load the .env file
    load_dotenv(dotenv_path=dotenv_path)
except:
    logging.warning(
        "Could not load .env file. Make sure it exists and python-dotenv is installed.")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FRED_API_KEY = os.environ.get("FRED_API_KEY", "")

# Series IDs of interest for market demand context
SERIES_IDS = {
    "unemployment_rate": "UNRATE",          # US Unemployment Rate
    "job_openings": "JTSJOL",               # Job Openings: Total Nonfarm
    "gdp_growth": "A191RL1Q225SBEA",        # Real GDP growth rate
    # Industrial production: semiconductors (proxy for tech sector)
    "tech_sector_index": "IPG3344S",
    # Consumer Price Index (inflation proxy)
    "cpi": "CPIAUCSL",
}


def fetch_series(fred, series_id, start="2018-01-01"):
    try:
        series = fred.get_series(series_id, observation_start=start)
        series.name = series_id
        return series
    except Exception as exc:
        logger.error("Failed to fetch series %s: %s", series_id, exc)
        return pd.Series(dtype=float, name=series_id)


def collect_economic_indicators(start="2018-01-01"):
    if not FRED_API_KEY:
        logger.warning("FRED_API_KEY not set. Returning empty DataFrame.")
        return pd.DataFrame()

    fred = Fred(api_key=FRED_API_KEY)
    series_list = []

    for name, series_id in SERIES_IDS.items():
        s = fetch_series(fred, series_id, start=start)
        s.name = name
        series_list.append(s)
        logger.info("Fetched %s (%s): %d observations",
                    name, series_id, len(s))

    df = pd.concat(series_list, axis=1)

    # Resample to monthly frequency (forward-fill quarterly/irregular series)
    df = df.resample("ME").ffill()
    return df


if __name__ == "__main__":
    PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    SAVE_PATH = f"{PARENT_DIR}/data/raw/economic_indicators.csv"

    econ_df = collect_economic_indicators()
    if not econ_df.empty:
        econ_df.to_csv(SAVE_PATH, index=False)
        logger.info("Saved economic indicators -> %s", SAVE_PATH)
    else:
        logger.warning("No economic data saved.")
