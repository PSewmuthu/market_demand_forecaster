# Market Demand Trend Analysis

A data pipeline for identifying emerging job roles, in-demand skills, and growing industries - built for outsourcing and recruitment market intelligence. The system combines job posting data, search-interest signals, and macroeconomic indicators to forecast demand trends and surface actionable hiring insights.

## Overview

This project addresses task **Market Demand Trend Analysis**: identifying emerging job roles, desired skills, and in-demand industries through job market data analysis, with time series forecasting to support workforce planning and recruitment strategy.

The pipeline covers seven stages:

1. **Data collection** — job postings, search-interest trends, and economic indicators
2. **NLP processing** — skill, role, and industry extraction from postings
3. **Seasonality analysis** — decomposition of historical demand patterns
4. **Forecasting** — time series models (Prophet, SARIMAX, LSTM) with external regressors
5. **Scenario analysis** — best-case, worst-case, and most-likely projections
6. **Validation** — backtesting against historical data
7. **Visualization** — 17 charts saved to `visualizations/`

## Features

- Multi-source data collection: Adzuna and Remotive job board APIs, Google Trends (via pytrends), and FRED economic indicators
- Keyword-based skill, role, and industry extraction with no external AI API dependency
- Seasonal decomposition (classical and STL methods) of demand patterns
- Forecasting with **Prophet**, **SARIMAX**, and **LSTM** (deep learning), all supporting external regressors (search trends, economic indicators)
- Scenario modeling that adjusts external factors to generate best-case, worst-case, and most-likely forecasts
- Model validation via holdout backtesting with MAE, RMSE, and MAPE metrics
- 17 publication-ready visualizations covering demand trends, forecasts, scenarios, backtests, and NLP insights
- Modular design - each stage runs standalone or as part of the full pipeline

## Project Structure

```
market_demand_forecast/
├── main.py                          # Pipeline orchestrator (7 steps)
├── requirements.txt
├── data_collection/
│   ├── pytrends_collector.py        # Google Trends search-interest data
│   ├── job_board_collector.py       # Adzuna + Remotive job postings
│   └── economic_collector.py        # FRED economic indicators
├── nlp/
│   └── skill_extraction.py          # Skill, role, and industry extraction
├── analysis/
│   ├── demand_target.py             # Builds demand proxy from Google Trends history
│   ├── revenue_seasonality.py       # Seasonal decomposition
│   ├── forecasting_models.py        # Prophet and SARIMAX models
│   ├── lstm_model.py                # LSTM deep learning forecasting model
│   ├── scenario_analysis.py         # Best/worst/most-likely scenarios
│   ├── validation.py                # Backtesting and accuracy metrics
│   └── visualize.py                 # All 17 visualizations
├── data/
│   ├── raw/                         # Collected raw data
│   └── processed/                   # NLP results, demand series, seasonality
├── predictions/
│   ├── forecasts/                   # Prophet, SARIMAX, LSTM, scenario forecasts
│   └── validation/                  # Backtest results and accuracy metrics
└── visualizations/                  # Generated charts
```

## Requirements

- Python 3.10+
- See `requirements.txt` for package dependencies

## Installation

```bash
git clone https://github.com/PSewmuthu/market_demand_forecaster.git
cd market_demand_forecaster
pip install -r requirements.txt
```

## Configuration

The pipeline uses three external data sources. Set the following environment variables to enable them:

| Variable         | Source                                                        | Required | Notes                |
| ---------------- | ------------------------------------------------------------- | -------- | -------------------- |
| `ADZUNA_APP_ID`  | [Adzuna API](https://developer.adzuna.com/)                   | Optional | Free tier available  |
| `ADZUNA_API_KEY` | Adzuna API                                                    | Optional | Free tier available  |
| `FRED_API_KEY`   | [FRED API](https://fred.stlouisfed.org/docs/api/api_key.html) | Optional | Free, instant signup |

Remotive's job board API requires no key and runs by default. If Adzuna or FRED credentials are not provided, those sources are skipped automatically and the pipeline continues with available data.

```bash
export ADZUNA_APP_ID="your_app_id"
export ADZUNA_API_KEY="your_api_key"
export FRED_API_KEY="your_fred_key"
```

## Usage

### Run the full pipeline

```bash
python main.py
```

This executes all seven stages in sequence and writes outputs to `data/raw/`, `data/processed/`, `predictions/forecasts/`, `predictions/validation/`, and `visualizations/`.

### Run individual stages

Each module can be run independently for development or debugging. Run them in this order since each stage depends on the previous one's output:

```bash
# Stage 1 – Data collection
python data_collection/pytrends_collector.py
python data_collection/job_board_collector.py
python data_collection/economic_collector.py

# Stage 2 – NLP processing
python nlp/skill_extraction.py

# Stage 3 – Demand series + seasonality
python analysis/demand_target.py
python analysis/revenue_seasonality.py

# Stage 4 – Forecasting (generates merged_features.csv + Prophet/SARIMAX forecasts)
python analysis/forecasting_models.py

# Stage 4 – LSTM forecast (reads merged_features.csv from stage 4 above)
python analysis/lstm_model.py

# Stage 5 – Scenario analysis (reads merged_features.csv)
python analysis/scenario_analysis.py

# Stage 6 – Validation / backtesting (reads merged_features.csv)
python analysis/validation.py

# Stage 7 – Visualizations (reads all processed outputs)
python analysis/visualize.py
```

## Output Files

### `data/processed/`

NLP results and demand analysis outputs:

| File                        | Description                                                                  |
| --------------------------- | ---------------------------------------------------------------------------- |
| `monthly_demand_series.csv` | Monthly demand proxy derived from Google Trends search-interest history      |
| `merged_features.csv`       | Demand series merged with external regressors (trends + economic indicators) |
| `jobs_with_skills.csv`      | Job postings enriched with extracted skills                                  |
| `top_skills.csv`            | Most frequently requested skills                                             |
| `emerging_roles.csv`        | Most common job titles by posting volume                                     |
| `top_industries.csv`        | Industries/categories ranked by demand                                       |
| `skill_trends_monthly.csv`  | Monthly time series of skill mentions                                        |
| `seasonality_by_month.csv`  | Average seasonal effect by calendar month                                    |

### `predictions/forecasts/`

Model forecast outputs:

| File                     | Description                                                 |
| ------------------------ | ----------------------------------------------------------- |
| `prophet_forecast.csv`   | Prophet model 12-month forecast with confidence intervals   |
| `sarimax_forecast.csv`   | SARIMAX model 12-month forecast with confidence intervals   |
| `lstm_forecast.csv`      | LSTM model 12-month forecast                                |
| `scenario_forecasts.csv` | Best-case, worst-case, and most-likely scenario projections |

### `predictions/validation/`

Backtest results and accuracy metrics:

| File                           | Description                                    |
| ------------------------------ | ---------------------------------------------- |
| `prophet_backtest.csv`         | Prophet actual vs predicted on 6-month holdout |
| `prophet_backtest_metrics.csv` | Prophet MAE, RMSE, MAPE                        |
| `sarimax_backtest.csv`         | SARIMAX actual vs predicted on 6-month holdout |
| `sarimax_backtest_metrics.csv` | SARIMAX MAE, RMSE, MAPE                        |
| `lstm_backtest.csv`            | LSTM actual vs predicted on 6-month holdout    |
| `lstm_backtest_metrics.csv`    | LSTM MAE, RMSE, MAPE                           |

### `visualizations/`

All 17 charts are generated automatically by `analysis/visualize.py` (step 7 of `main.py`). Run `python analysis/visualize.py` standalone at any time to regenerate them.

| File                            | Description                                           |
| ------------------------------- | ----------------------------------------------------- |
| `01_demand_overview.png`        | Historical Google Trends demand index with area fill  |
| `02_seasonal_decomposition.png` | Observed / trend / seasonal / residual breakdown      |
| `03_seasonality_by_month.png`   | Average seasonal effect by calendar month (bar chart) |
| `04_prophet_forecast.png`       | Prophet fitted values + 12-month forecast with 95% CI |
| `05_sarimax_forecast.png`       | SARIMAX 12-month forecast with confidence intervals   |
| `06_lstm_forecast.png`          | LSTM 12-month forecast                                |
| `07_forecast_comparison.png`    | All three models overlaid on one chart                |
| `08_scenario_analysis.png`      | Best / most-likely / worst-case scenario projections  |
| `09_prophet_backtest.png`       | Prophet: actual vs predicted on 6-month holdout       |
| `10_sarimax_backtest.png`       | SARIMAX: actual vs predicted on 6-month holdout       |
| `11_lstm_backtest.png`          | LSTM: actual vs predicted on 6-month holdout          |
| `12_backtest_comparison.png`    | All three backtests side by side                      |
| `13_metrics_comparison.png`     | MAE / RMSE / MAPE bar chart per model                 |
| `14_top_skills.png`             | Top 15 in-demand skills (horizontal bar chart)        |
| `15_emerging_roles.png`         | Top 15 emerging job roles by posting volume           |
| `16_top_industries.png`         | Industry distribution pie chart                       |
| `17_skill_trends_heatmap.png`   | Monthly skill demand heatmap (skills × months)        |

## Limitations

- Google Trends data via `pytrends` is rate-limited and returns a relative search-interest index (0–100), not absolute volume.
- Job board coverage depends on API access (Adzuna requires registration; Remotive is open but remote-job focused). Job posting "created" dates reflect currently-live listings only, not historical posting volume.
- Forecast accuracy depends on data volume - at least 24 months of monthly data is recommended for reliable seasonal decomposition and forecasting; 60+ months is preferable for the LSTM model.
- Future regressor values in forecasts are carried forward from the last observed value as a naive assumption; replace with dedicated forecasts of those regressors for improved accuracy.
- With only ~5 years of Trends history and limited regressors, SARIMAX may occasionally fail to fully converge; this is a known limitation of fitting seasonal ARIMA models to short series.
- TensorFlow (required for the LSTM model) needs AVX/AVX2 CPU instructions. If your CPU lacks these, install `tensorflow-cpu==2.13.0` or `tensorflow==2.10.0` for broader compatibility. If TensorFlow cannot be loaded, the pipeline automatically skips the LSTM step and continues with Prophet and SARIMAX.
