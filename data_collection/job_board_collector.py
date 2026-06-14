"""
job_board_collector.py
-----------------------
Collects job posting data from:
  - Adzuna API (requires free APP_ID + API_KEY: https://developer.adzuna.com/)
  - Remotive API (no key required, remote/outsourcing-focused jobs)

Output: raw job postings dataframe with title, category, company,
salary range, posted date, and description (for later skill extraction).
"""

import os
import time
import logging
import requests
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- Configuration ---
ADZUNA_APP_ID = os.environ.get("ADZUNA_APP_ID", "")
ADZUNA_API_KEY = os.environ.get("ADZUNA_API_KEY", "")
ADZUNA_COUNTRY = "gb"  # gb, us, au, etc.

ADZUNA_CATEGORIES = [
    "it-jobs",
    "engineering-jobs",
    "scientific-qa-jobs",
    "accounting-finance-jobs",
]

REMOTIVE_URL = "https://remotive.com/api/remote-jobs"


def fetch_adzuna_jobs(category, pages=5, results_per_page=50, country=ADZUNA_COUNTRY):
    """Fetch job postings from Adzuna for a given category."""
    if not ADZUNA_APP_ID or not ADZUNA_API_KEY:
        logger.warning(
            "Adzuna credentials not set. Skipping Adzuna fetch for %s.", category)
        return pd.DataFrame()

    records = []
    for page in range(1, pages + 1):
        url = f"https://api.adzuna.com/v1/api/jobs/{country}/search/{page}"
        params = {
            "app_id": ADZUNA_APP_ID,
            "app_key": ADZUNA_API_KEY,
            "category": category,
            "results_per_page": results_per_page,
            "content-type": "application/json",
        }
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            logger.error("Adzuna request failed (page %d, %s): %s",
                         page, category, exc)
            break

        results = data.get("results", [])
        if not results:
            break

        for job in results:
            records.append({
                "source": "adzuna",
                "title": job.get("title"),
                "category": job.get("category", {}).get("label"),
                "company": job.get("company", {}).get("display_name"),
                "location": job.get("location", {}).get("display_name"),
                "salary_min": job.get("salary_min"),
                "salary_max": job.get("salary_max"),
                "created": job.get("created"),
                "description": job.get("description", ""),
                "redirect_url": job.get("redirect_url"),
            })

        time.sleep(1)  # be polite

    logger.info("Fetched %d jobs from Adzuna (%s)", len(records), category)
    return pd.DataFrame(records)


def fetch_remotive_jobs(search=None, limit=None):
    """Fetch remote/outsourcing job postings from Remotive (no key required)."""
    params = {}
    if search:
        params["search"] = search
    if limit:
        params["limit"] = limit

    try:
        resp = requests.get(REMOTIVE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        logger.error("Remotive request failed: %s", exc)
        return pd.DataFrame()

    jobs = data.get("jobs", [])
    records = []
    for job in jobs:
        records.append({
            "source": "remotive",
            "title": job.get("title"),
            "category": job.get("category"),
            "company": job.get("company_name"),
            "location": job.get("candidate_required_location"),
            "salary_min": None,
            "salary_max": None,
            "created": job.get("publication_date"),
            "description": job.get("description", ""),
            "redirect_url": job.get("url"),
        })

    logger.info("Fetched %d jobs from Remotive", len(records))
    return pd.DataFrame(records)


def collect_all_jobs():
    """Collect jobs from all configured sources and merge into one DataFrame."""
    frames = []

    for category in ADZUNA_CATEGORIES:
        df = fetch_adzuna_jobs(category)
        if not df.empty:
            frames.append(df)

    remotive_df = fetch_remotive_jobs()
    if not remotive_df.empty:
        frames.append(remotive_df)

    if not frames:
        logger.warning("No job data collected from any source.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined["created"] = pd.to_datetime(
        combined["created"], errors="coerce", utc=True)
    combined = combined.dropna(subset=["title", "created"])
    return combined


if __name__ == "__main__":
    PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    SAVE_PATH = f"{PARENT_DIR}/data/raw/job_postings.csv"

    jobs_df = collect_all_jobs()
    if not jobs_df.empty:
        jobs_df.to_csv(SAVE_PATH, index=False)
        logger.info(
            "Saved %d total job postings -> %s", len(jobs_df), SAVE_PATH)
    else:
        logger.warning("No data saved.")
