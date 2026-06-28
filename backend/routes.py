import json
import os
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional

from models.predict import predictor
from models.risk_scorer import risk_scorer
from agents.weather_agent import WeatherAgent
from agents.soil_agent import SoilAgent
from agents.history_agent import HistoryAgent
from agents.risk_agent import RiskAgent
from genai.explainer import explainer

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

def to_json(data):
    return json.dumps(data, cls=NumpyEncoder, ensure_ascii=False)

router = APIRouter()

class PredictionRequest(BaseModel):
    district: str
    crop: str
    year: int = 2024
    month: int = 6
    sowing_area_ha: float = 100.0
    irrigation_pct: float = 60.0
    fertilizer_kg_per_ha: float = 150.0
    pesticide_kg_per_ha: float = 10.0
    avg_temp_c: Optional[float] = None
    rainfall_mm: Optional[float] = None
    humidity_pct: Optional[float] = None
    sunshine_hours: Optional[float] = None
    wind_speed_kmh: Optional[float] = None
    N_mg_per_kg: Optional[float] = None
    P_mg_per_kg: Optional[float] = None
    K_mg_per_kg: Optional[float] = None
    pH: Optional[float] = None
    organic_carbon_pct: Optional[float] = None
    avg_yield_5yr: Optional[float] = None
    pest_history: str = "low"
    drought_history: str = "none"
    soil_type: str = "Alluvial"

weather_agent = WeatherAgent()
soil_agent = SoilAgent()
history_agent = HistoryAgent()
risk_agent = RiskAgent()

@router.get("/health")
def health():
    return Response(to_json({"status": "healthy", "service": "YieldPredict AI"}), media_type="application/json")

@router.get("/crops")
def get_crops():
    try:
        df = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "crop_yield_data.csv"))
        crops = sorted(df['crop'].unique().tolist())
    except:
        crops = ["Rice", "Wheat", "Maize", "Cotton", "Sugarcane", "Groundnut", "Soybean"]
    return Response(to_json({"crops": crops, "total": len(crops)}), media_type="application/json")

@router.get("/districts")
def get_districts():
    try:
        df = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "crop_yield_data.csv"))
        districts = sorted(df['district'].unique().tolist())
    except:
        districts = ["Guntur", "Krishna", "East Godavari", "West Godavari", "Prakasam"]
    return Response(to_json({"districts": districts, "total": len(districts)}), media_type="application/json")

@router.get("/models/leaderboard")
def get_leaderboard():
    leaderboard_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "saved_models", "model_leaderboard.json")
    if not os.path.exists(leaderboard_path):
        return Response(to_json({"models": [], "message": "No leaderboard data available"}), media_type="application/json")
    with open(leaderboard_path) as f:
        data = json.load(f)
    best = max(data, key=lambda x: x.get('R\u00b2', 0) if isinstance(x.get('R\u00b2'), (int, float)) else 0)
    return Response(to_json({"models": data, "best_model": best["Model"], "best_r2": best.get('R\u00b2', 0)}), media_type="application/json")

@router.post("/predict")
def predict_yield(req: PredictionRequest):
    try:
        input_data = req.model_dump()
        input_data = {k: (v if v is not None else 0) for k, v in input_data.items()}

        weather_info = weather_agent.analyze(input_data)
        soil_info = soil_agent.analyze(input_data)
        history_info = history_agent.analyze(input_data)

        # Merge weather data into context (use actual values, not forecast)
        for k in ["avg_temp_c", "rainfall_mm", "humidity_pct", "sunshine_hours", "wind_speed_kmh"]:
            if k in weather_info and (input_data.get(k) is None or input_data.get(k) == 0):
                input_data[k] = weather_info[k]

        # Merge soil data into context
        for k in ["N_mg_per_kg", "P_mg_per_kg", "K_mg_per_kg", "pH", "organic_carbon_pct",
                   "zinc_mg_per_kg", "iron_mg_per_kg", "copper_mg_per_kg", "manganese_mg_per_kg",
                   "boron_mg_per_kg", "soil_type"]:
            if k in soil_info and (input_data.get(k) is None or input_data.get(k) == 0):
                val = soil_info[k]
                if isinstance(val, str):
                    try:
                        val = float(val.split()[0])
                    except:
                        val = input_data.get(k, 0)
                if isinstance(val, (int, float)):
                    input_data[k] = val

        input_data["avg_yield_5yr"] = history_info.get("avg_yield_5yr", 3.0)
        input_data["pest_history"] = history_info.get("pest_history", "low")
        input_data["drought_history"] = history_info.get("drought_history", "none")
        input_data["area_trend"] = history_info.get("area_trend", "stable")

        model_info = predictor.get_model_info(req.crop)

        yield_pred = predictor.predict(input_data)

        risk_info = risk_agent.analyze(input_data)

        stage_preds = {}
        for days in [30, 60, 90, 120, 150]:
            stage_preds[f"day_{days}"] = round(predictor.predict_stage(input_data, days), 3)

        explanation = explainer.explain(
            input_data, yield_pred, risk_info, weather_info, soil_info, history_info
        )

        result = {
            "status": "success",
            "district": req.district,
            "crop": req.crop,
            "model_info": model_info,
            "yield_prediction": {
                "value_t_per_ha": round(yield_pred, 3),
                "unit": "tonnes per hectare"
            },
            "stage_predictions": stage_preds,
            "weather_analysis": weather_info,
            "soil_analysis": soil_info,
            "historical_analysis": history_info,
            "risk_assessment": risk_info,
            "explanation": explanation
        }
        return Response(to_json(result), media_type="application/json")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

@router.post("/predict/full")
def predict_full(req: PredictionRequest):
    return predict_yield(req)

@router.post("/simulate/climate")
def simulate_climate(district: str = "Guntur", crop: str = "Rice"):
    input_data = {
        "district": district, "crop": crop, "year": 2024, "month": 6,
        "sowing_area_ha": 100, "irrigation_pct": 60, "fertilizer_kg_per_ha": 150,
        "pesticide_kg_per_ha": 10, "pest_history": "low", "drought_history": "none"
    }
    yield_pred = predictor.predict(input_data)
    y_std = 0.15 * yield_pred

    base = risk_scorer.monte_carlo_risk(yield_pred, y_std)

    scenarios = {}
    for temp_rise in [1.0, 2.0, 3.0]:
        for rain_change in [-0.3, -0.1, 0.1]:
            adjusted_yield = yield_pred * (1 - 0.05 * temp_rise + 0.2 * rain_change)
            mc = risk_scorer.monte_carlo_risk(max(0.1, adjusted_yield), y_std)
            label = f"+{temp_rise}°C_{rain_change*100:+.0f}%rain"
            scenarios[label] = {
                "yield": round(max(0.1, adjusted_yield), 3),
                "default_probability": mc["default_probability"],
                "expected_loss": mc["expected_loss"]
            }

    result = {
        "base_yield": round(yield_pred, 3),
        "base_risk": base,
        "scenarios": scenarios
    }
    return Response(to_json(result), media_type="application/json")
