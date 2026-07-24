# 🌾 YieldPredict AI

Crop yield forecasting and insurance risk analysis for Indian agriculture, built entirely on real government and satellite data — no synthetic records anywhere in the pipeline.

The motivation was straightforward: crop insurance in India is hard to price fairly because yield risk varies enormously by district, crop, and season, and a lot of existing tools either oversimplify (one model for every crop) or rely on data that isn't actually grounded in what happened on the ground. So this project tries to do both things properly — predict yield per crop with whichever model actually works best for that crop, and turn that into a risk/premium estimate using a method that doesn't change its answer every time you run it.

## What it actually does

Given a district, crop, and season, the system pulls real weather (NASA POWER), real soil characteristics (ISRIC SoilGrids + the Indian Soil Health Survey), and real historical yield trends (Government of India DES records going back to 1997), feeds all of that into a crop-specific model, and produces both a yield prediction and an insurance risk score.

## Why per-crop model routing

Early versions of this used a single model for every crop, and the results were mediocre across the board — which makes sense in hindsight, since cotton yield behaves nothing like wheat yield statistically. So instead I trained and compared 11 different ML models per crop and let each crop pick its own best performer:

- **Cotton** → RandomForest (R² = 0.977)
- **Wheat** → XGBoost (R² = 0.899)
- ...and so on per crop, with the full leaderboard logged so you can see why each one won.

Overall, **XGBoost comes out on top across crops on average** (R² = 0.956), but it's not universally the best — that's the whole point of routing per crop instead of forcing one model on everything.

| Model | R² | MAE |
|---|---|---|
| XGBoost | 0.9562 | 0.659 |
| RandomForest | 0.9545 | 0.603 |
| GradientBoosting | 0.9518 | 0.680 |
| MLPRegressor | 0.9477 | 0.707 |
| Ridge / Lasso / BayesianRidge | ~0.926 | ~0.803 |

## Why the risk scoring is deterministic

This is the design decision I care about most in this repo. The risk/premium component uses Beta-Binomial posterior inference — but deliberately **without** random sampling. I tested with Monte Carlo, but the results were not deterministic and were slow.

The reasoning: if this is ever going to inform an actual premium number for an actual farmer, it can't give a slightly different answer every time someone re-runs it on the same inputs. A stochastic risk score is fine for a research notebook, but it's a real problem for anything resembling an audit trail or a regulatory conversation. Computing the posterior in closed form means the same district + crop + season always produces the exact same risk score — reproducible by design, not just by coincidence.

## The agent system

Rather than have one big function try to gather everything, there are four small agents that each own one piece of context and fetch real data before a prediction happens:

- **Weather Agent** — pulls actual NASA POWER data for the relevant location/season (temperature, rainfall, humidity, wind, sunlight)
- **Soil Agent** — pulls real ISRIC SoilGrids data plus state-wise medians from the Indian Soil Health Survey for pH, nitrogen, organic carbon, P/K and micronutrients
- **History Agent** — looks at the actual historical yield trend for that district and crop
- **Risk Agent** — takes everything the other three agents found and runs the Bayesian risk + premium scoring

Splitting it this way also made debugging much easier — if a prediction looks off, I can check exactly which agent's data is being used instead of digging through one monolithic pipeline.

## Why "zero synthetic data" is a hard rule

There's no fallback dummy data anywhere in this codebase. If NASA POWER or ISRIC doesn't have data for a given district/date, the pipeline says so rather than quietly filling in a plausible-looking number. For a tool that's ultimately trying to estimate insurance risk, a fabricated soil reading that *looks* reasonable is worse than an honest gap — it's the kind of thing that erodes trust the moment someone checks your work.

## The data behind it

| Source | What it provides | Records |
|---|---|---|
| Govt of India DES (via HuggingFace) | District-wise crop yields | 235,404 |
| NASA POWER API | Weather — temp, rain, humidity, wind, sun | 143,628 |
| ISRIC SoilGrids API | Soil pH, nitrogen, organic carbon | 630 |
| Indian Soil Health Survey | P, K, micronutrients (state-wise medians) | 630 |

The yield data spans **646 districts and 77 crops from 1997–2015** — long enough to actually capture multi-year trend and variability per crop, rather than just a couple of seasons.

## Architecture

```
backend/routes.py            → FastAPI endpoints
agents/weather_agent.py      → Real NASA POWER data lookup
agents/soil_agent.py         → Real ISRIC + Indian survey data
agents/history_agent.py      → Historical yield trends
agents/risk_agent.py         → Bayesian risk + premium scoring
models/predict.py            → Per-crop model router
models/risk_scorer.py        → Deterministic Beta-Binomial scoring
data/real_data_pipeline.py   → Fetch + cache real data
frontend/                    → Dashboard (Chart.js)
```

## Dashboard

The frontend gives you a single place to explore a prediction rather than just a raw number:

- **Prediction** — yield forecast for the chosen district/crop/season
- **Stage-wise charts** — how the crop's growth stages map to the prediction
- **Climate simulation** — see how the forecast shifts if you nudge weather inputs
- **Premium estimation** — the Bayesian risk score translated into an actual premium figure
- **Model leaderboard** — the full per-crop model comparison, not just the winner

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Fetches real data and trains all per-crop models — takes a while the first time
python3 main.py --setup-only

# Start the server
python3 main.py --port 8000
```

Then visit:
- Dashboard → `http://localhost:8000/dashboard`
- API → `http://localhost:8000/api`
- Docs → `http://localhost:8000/docs`

A heads-up: `--setup-only` is doing real work — fetching and caching government, NASA, and ISRIC data plus training 11 models per crop — so don't expect it to finish in seconds. Subsequent runs are much faster since the data gets cached locally.

## Known limitations

- Yield history runs through 2015 — there's a real gap to the present that newer government data releases could close, but I haven't integrated anything more recent yet.
- District-level granularity means within-district variation (a farmer in a particularly dry pocket of a district, for instance) isn't captured.
- The risk/premium output is a research-grade estimate, not an actuarially certified number — it shouldn't be used to actually underwrite insurance without review by someone qualified to do that.

## License

MIT — see `LICENSE` for the full text. In short: use it, modify it, build on it, just don't treat the risk scores here as a substitute for actual actuarial sign-off if you're doing anything with real money or real farmers.
