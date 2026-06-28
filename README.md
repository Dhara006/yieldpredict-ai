# 🌾 YieldPredict AI

Crop yield forecasting and insurance risk analyzer for Indian agriculture. Uses real data from Government of India, NASA POWER, and ISRIC SoilGrids — zero synthetic data.

## Features

- **11 ML models** compared (XGBoost best: R²=0.956)
- **Per-crop model routing** — each crop gets its best model (RandomForest for Cotton R²=0.977, XGBoost for Wheat R²=0.899, etc.)
- **Deterministic Bayesian risk scoring** — Beta-Binomial posterior inference, no random sampling
- **Real data pipeline** — Govt of India DES yields (235K records, 646 districts, 77 crops, 1997–2015), NASA POWER weather, ISRIC soil
- **Agent system** — Weather, Soil, History, Risk agents auto-fill real context before prediction
- **Dashboard** — Prediction, stage-wise charts, climate simulation, premium estimation, model leaderboard

## Quick Start

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Generate data & train models
python3 main.py --setup-only

# Start server
python3 main.py --port 8000
# Dashboard: http://localhost:8000/dashboard
# API:       http://localhost:8000/api
# Docs:      http://localhost:8000/docs
```

## Data Sources

| Source | Data | Records |
|--------|------|---------|
| Govt of India DES (via HuggingFace) | District-wise crop yields | 235,404 |
| NASA POWER API | Weather (temp, rain, humidity, wind, sun) | 143,628 |
| ISRIC SoilGrids API | Soil pH, nitrogen, organic carbon | 630 |
| Indian Soil Health Survey | P, K, micronutrients (state-wise medians) | 630 |

## Architecture

```
backend/routes.py       →  FastAPI endpoints
agents/weather_agent.py →  Real NASA POWER data lookup
agents/soil_agent.py    →  Real ISRIC + Indian survey data
agents/history_agent.py →  Historical yield trends
agents/risk_agent.py    →  Bayesian risk + premium
models/predict.py       →  Per-crop model router
models/risk_scorer.py   →  Deterministic Beta-Binomial scoring
data/real_data_pipeline.py → Fetch + cache real data
frontend/               →  Dashboard with Chart.js
```

## Model Leaderboard

| Model | R² | MAE |
|-------|-----|-----|
| XGBoost | 0.9562 | 0.659 |
| RandomForest | 0.9545 | 0.603 |
| GradientBoosting | 0.9518 | 0.680 |
| MLPRegressor | 0.9477 | 0.707 |
| Ridge/Lasso/BayesianRidge | ~0.926 | ~0.803 |

## License

MIT
