"""
skill_extraction.py
---------------------
Extracts skills, role categories, and industry signals from job posting
titles and descriptions using keyword matching (no AI API required).

Designed to feed:
  - top in-demand skills
  - emerging job roles
  - in-demand industries
"""

import re
import logging
from collections import Counter
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# Extendable skill dictionary -- lowercase, word-boundary matched
SKILL_DICT = [
    # Programming / technical
    "python", "java", "javascript", "typescript", "sql", "nosql", "c++", "go", "rust",
    "react", "angular", "vue", "node.js", "django", "flask",
    # Data / ML
    "machine learning", "deep learning", "data analysis", "data engineering",
    "tensorflow", "pytorch", "pandas", "numpy", "tableau", "power bi", "excel",
    "nlp", "computer vision", "llm", "generative ai",
    # Cloud / infra
    "aws", "azure", "gcp", "cloud computing", "docker", "kubernetes", "terraform",
    "devops", "ci/cd", "linux",
    # Soft / business
    "project management", "agile", "scrum", "stakeholder management",
    "communication", "leadership", "problem solving",
    # Marketing / other
    "digital marketing", "seo", "content marketing", "social media marketing",
    "cybersecurity", "blockchain", "ux design", "ui design",
]


def extract_skills(text, skill_dict=None):
    """Return list of skills found in a text (case-insensitive, word-boundary)."""

    if not isinstance(text, str):
        return []

    skill_dict = skill_dict or SKILL_DICT
    text_lower = text.lower()
    found = []

    for skill in skill_dict:
        pattern = r"\b" + re.escape(skill) + r"\b"

        if re.search(pattern, text_lower):
            found.append(skill)

    return found


def add_skill_column(jobs_df, text_col="description", title_col="title"):
    """Add a 'skills' column by combining title + description text."""

    combined_text = jobs_df[title_col].fillna(
        "") + " " + jobs_df[text_col].fillna("")
    jobs_df = jobs_df.copy()

    jobs_df["skills"] = combined_text.apply(extract_skills)
    jobs_df["skill_count"] = jobs_df["skills"].apply(len)

    return jobs_df


def top_skills(jobs_df, n=15):
    """Return top-n skills by frequency across all postings."""

    all_skills = [s for skills in jobs_df["skills"] for s in skills]

    return pd.Series(Counter(all_skills)).sort_values(ascending=False).head(n)


def emerging_roles(jobs_df, title_col="title", n=15):
    """Return top-n most common job titles (proxy for emerging roles)."""

    cleaned = jobs_df[title_col].str.strip().str.lower()

    return cleaned.value_counts().head(n)


def top_industries(jobs_df, category_col="category", n=10):
    """Return top-n industries/categories by posting volume."""

    return jobs_df[category_col].value_counts().head(n)


def skill_trend_by_month(jobs_df, date_col="created"):
    """
    Build a monthly time series of skill mention counts.
    Returns a wide DataFrame: index=month, columns=skill, values=count.
    """

    df = jobs_df.copy()
    df[date_col] = pd.to_datetime(
        df[date_col], errors="coerce", utc=True).dt.tz_localize(None)
    df = df.dropna(subset=[date_col])
    df["month"] = df[date_col].dt.to_period("M").dt.to_timestamp()

    exploded = df.explode("skills").dropna(subset=["skills"])
    pivot = exploded.pivot_table(
        index="month", columns="skills", values="title", aggfunc="count", fill_value=0
    )

    return pivot


if __name__ == "__main__":
    import os

    PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_RAW = os.path.join(PARENT_DIR, "data", "raw")
    DATA_PROCESSED = os.path.join(PARENT_DIR, "data", "processed")

    os.makedirs(DATA_PROCESSED, exist_ok=True)

    jobs_df = pd.read_csv(os.path.join(DATA_RAW, "job_postings.csv"))
    jobs_df = add_skill_column(jobs_df)

    logger.info("Top skills:\n%s", top_skills(jobs_df))
    logger.info("Emerging roles:\n%s", emerging_roles(jobs_df))
    logger.info("Top industries:\n%s", top_industries(jobs_df))

    jobs_df.to_csv(os.path.join(
        DATA_PROCESSED, "jobs_with_skills.csv"), index=False)

    skill_trends = skill_trend_by_month(jobs_df)
    skill_trends.to_csv(os.path.join(
        DATA_PROCESSED, "skill_trends_monthly.csv"))
    logger.info("Saved processed job data and skill trends.")
