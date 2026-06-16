"""
demand_target.py
-------------------
Builds the primary demand proxy series for forecasting from Google Trends
search-interest data (trends_roles.csv / trends_skills.csv / trends_industries.csv).

Background:
  Job board snapshots (Adzuna/Remotive) only return *currently live* postings,
  so a "monthly posting count" derived from `created` dates is dominated by
  whatever was posted in the last 1-3 months (most postings expire quickly).
  That makes it a snapshot artifact, not a real historical time series.

  Google Trends data, by contrast, is genuinely longitudinal -- each pull
  returns a multi-year weekly history of search interest, which is a
  legitimate (if imperfect) proxy for market demand over time.

This module:
  - Loads one or more trends CSVs (roles/skills/industries)
  - Aggregates selected columns into a single demand index
  - Resamples weekly -> monthly
  - Returns a 'y' series suitable for seasonality/forecasting modules
"""

import logging
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_trends_csv(path):
    """Load a trends CSV with a date index."""

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()]
    df = df[~df.index.duplicated(keep="last")].sort_index()

    return df


def build_demand_index(trends_dfs, columns=None, agg="mean"):
    """
    Combine one or more trends DataFrames into a single demand index series.

    trends_dfs: dict of {name: DataFrame} (e.g. {"roles": roles_df, "skills": skills_df})
    columns: dict of {name: [list of columns]} to select from each DataFrame.
             If None, all columns from each DataFrame are used.
    agg: 'mean' or 'sum' -- how to combine the selected columns into one index.

    Returns a single pd.Series named 'y', weekly frequency (same as input).
    """

    columns = columns or {}
    pieces = []

    for name, df in trends_dfs.items():
        cols = columns.get(name, df.columns.tolist())
        cols = [c for c in cols if c in df.columns]
        if not cols:
            logger.warning("No matching columns for '%s', skipping.", name)
            continue
        pieces.append(df[cols])

    if not pieces:
        raise ValueError(
            "No valid columns found across provided trends DataFrames.")

    combined = pd.concat(pieces, axis=1)

    if agg == "sum":
        demand = combined.sum(axis=1)
    else:
        demand = combined.mean(axis=1)

    demand.name = "y"
    return demand


def weekly_to_monthly(series, agg="mean"):
    """Resample a weekly series to month-end frequency."""

    if agg == "sum":
        return series.resample("ME").sum()

    return series.resample("ME").mean()


def build_monthly_demand_series(trends_paths, columns=None, agg="mean", monthly_agg="mean"):
    """
    End-to-end: load trends CSVs, build a combined demand index, resample to monthly.

    trends_paths: dict of {name: filepath}, e.g.
        {"roles": "../data/raw/trends_roles.csv",
         "skills": "../data/raw/trends_skills.csv",
         "industries": "../data/raw/trends_industries.csv"}
    columns: optional dict of {name: [columns]} to restrict which series contribute
    agg: how to combine columns within/across trends DataFrames ('mean' or 'sum')
    monthly_agg: how to resample weekly -> monthly ('mean' or 'sum')

    Returns: pd.Series named 'y', monthly (month-end) frequency.
    """

    trends_dfs = {}
    for name, path in trends_paths.items():
        try:
            trends_dfs[name] = load_trends_csv(path)
            logger.info("Loaded %s: %d rows, %s to %s",
                        name, len(trends_dfs[name]),
                        trends_dfs[name].index.min().date(),
                        trends_dfs[name].index.max().date())
        except FileNotFoundError:
            logger.warning("File not found: %s, skipping '%s'.", path, name)

    if not trends_dfs:
        raise FileNotFoundError(
            "None of the provided trends files were found.")

    weekly_demand = build_demand_index(trends_dfs, columns=columns, agg=agg)
    monthly_demand = weekly_to_monthly(weekly_demand, agg=monthly_agg)
    monthly_demand.name = "y"

    return monthly_demand


if __name__ == "__main__":
    import os

    PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_RAW = os.path.join(PARENT_DIR, "data", "raw")
    DATA_PROCESSED = os.path.join(PARENT_DIR, "data", "processed")

    trends_paths = {
        "roles": os.path.join(DATA_RAW, "trends_roles.csv"),
        "skills": os.path.join(DATA_RAW, "trends_skills.csv"),
        "industries": os.path.join(DATA_RAW, "trends_industries.csv"),
    }

    monthly_demand = build_monthly_demand_series(
        trends_paths, agg="mean", monthly_agg="mean")
    monthly_demand.to_csv(os.path.join(
        DATA_PROCESSED, "monthly_demand_series.csv"))
    logger.info(
        "Saved monthly demand series (%d months, %s to %s) -> %s",
        len(monthly_demand), monthly_demand.index.min(
        ).date(), monthly_demand.index.max().date(),
        os.path.join(DATA_PROCESSED, "monthly_demand_series.csv")
    )
