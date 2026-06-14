"""
pytrends_collector.py
----------------------
Collects Google Trends search-interest time series for job roles,
skills, and industries. Acts as a demand "signal" dataset.

Note: pytrends is unofficial and rate-limited. Keep batches <=5 keywords,
add delays between requests.
"""

import time
import logging
import pandas as pd
from pathlib import Path
from pytrends.request import TrendReq

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# Group keywords in batches of <=5 (pytrends hard limit)
KEYWORD_GROUPS = {
    "roles": [
        ["python developer", "data analyst", "cybersecurity analyst",
            "AI engineer", "cloud architect"],
        ["devops engineer", "ux designer", "product manager",
            "data scientist", "full stack developer"]
    ],
    "skills": [
        ["sql", "machine learning", "cloud computing",
            "project management", "digital marketing"],
        ["docker", "kubernetes", "react", "tableau", "agile"]
    ],
    "industries": [
        ["outsourcing services", "remote work",
            "fintech", "healthcare it", "e-commerce"]
    ]
}


def fetch_trend_group(pytrends, keywords, timeframe="today 5-y", geo="", retries=3, delay=60):
    """Fetch interest-over-time for a single keyword group with retry logic."""

    for attempt in range(1, retries + 1):
        try:
            pytrends.build_payload(keywords, timeframe=timeframe, geo=geo)
            df = pytrends.interest_over_time()

            if df.empty:
                logger.warning("Empty result for %s", keywords)
                return pd.DataFrame()

            df = df.drop(columns=["isPartial"], errors="ignore")
            logger.info("Fetched: %s", keywords)

            return df
        except Exception as exc:
            logger.warning("Attempt %d failed for %s: %s",
                           attempt, keywords, exc)
            time.sleep(delay)

    logger.error("All retries failed for %s", keywords)
    return pd.DataFrame()


def collect_all_trends(timeframe="today 5-y", geo="", delay_between=60):
    """Collect trends for all categories and merge into one DataFrame per category."""

    pytrends = TrendReq(hl="en-US", tz=0)
    results = {}

    for category, groups in KEYWORD_GROUPS.items():
        frames = []
        for group in groups:
            df = fetch_trend_group(
                pytrends, group, timeframe=timeframe, geo=geo)
            if not df.empty:
                frames.append(df)
            time.sleep(delay_between)

        if frames:
            merged = pd.concat(frames, axis=1)
            merged = merged.loc[:, ~merged.columns.duplicated()]
            results[category] = merged
        else:
            results[category] = pd.DataFrame()

    return results


def save_trends(results, output_dir="data/raw"):
    """Save each category's trends DataFrame to CSV files."""

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for category, df in results.items():
        if df.empty:
            logger.warning("Skipping save for empty category: %s", category)
            continue

        path = f"{output_dir}/trends_{category}.csv"
        df.to_csv(path)
        logger.info("Saved %s -> %s", category, path)


if __name__ == "__main__":
    import os

    PARENT_DIR = os.path.dirname(os.path.dirname(__file__))

    trends = collect_all_trends(timeframe="today 5-y", geo="")
    save_trends(trends, output_dir=f"{PARENT_DIR}/data/raw")
