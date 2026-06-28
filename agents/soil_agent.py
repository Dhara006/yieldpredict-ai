import pandas as pd
import os
from .base_agent import BaseAgent

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

class SoilAgent(BaseAgent):
    def __init__(self):
        super().__init__("Soil Agent")
        self.soil_data = pd.read_csv(os.path.join(DATA_DIR, "soil_data.csv"))

    def analyze(self, context):
        district = context.get("district", "")
        crop = context.get("crop", "Wheat")

        district_soil = self.soil_data[self.soil_data["district"] == district]
        if district_soil.empty:
            district_soil = self.soil_data.iloc[[0]]
        soil = district_soil.iloc[0]

        N, P, K = soil["N_mg_per_kg"], soil["P_mg_per_kg"], soil["K_mg_per_kg"]
        pH = soil["pH"]
        oc = soil["organic_carbon_pct"]
        soil_type = soil["soil_type"]

        n_status = "deficient" if N < 100 else "adequate" if N < 200 else "sufficient"
        p_status = "deficient" if P < 30 else "adequate" if P < 60 else "sufficient"
        k_status = "deficient" if K < 100 else "adequate" if K < 180 else "sufficient"
        ph_status = "acidic" if pH < 6.0 else "neutral" if pH < 7.5 else "alkaline"
        oc_status = "low" if oc < 0.5 else "moderate" if oc < 1.0 else "high"

        crop_soil_req = {
            "Rice": {"ph": (5.0, 7.5), "n": "high"},
            "Wheat": {"ph": (6.0, 7.5), "n": "high"},
            "Maize": {"ph": (5.5, 7.5), "n": "high"},
            "Cotton(lint)": {"ph": (5.5, 8.0), "n": "medium"},
            "Sugarcane": {"ph": (6.0, 7.5), "n": "high"},
            "Groundnut": {"ph": (5.5, 7.0), "n": "low"},
            "Soyabean": {"ph": (6.0, 7.0), "n": "medium"},
        }
        req = crop_soil_req.get(crop, {"ph": (5.5, 7.5), "n": "medium"})
        ph_fit = req["ph"][0] <= pH <= req["ph"][1]
        suitability = "excellent" if ph_fit and n_status != "deficient" else "good" if ph_fit else "poor"

        recommendations = []
        if n_status == "deficient": recommendations.append("Apply nitrogen fertilizer")
        if p_status == "deficient": recommendations.append("Apply phosphatic fertilizer")
        if k_status == "deficient": recommendations.append("Apply potassic fertilizer")
        if ph_status == "acidic": recommendations.append("Apply lime to raise pH")
        elif ph_status == "alkaline": recommendations.append("Apply gypsum")
        if oc_status == "low": recommendations.append("Add farmyard manure")

        return {
            "soil_type": soil_type,
            "N_status": f"{N:.1f} mg/kg ({n_status})",
            "P_status": f"{P:.1f} mg/kg ({p_status})",
            "K_status": f"{K:.1f} mg/kg ({k_status})",
            "pH": float(pH),
            "pH_status": ph_status,
            "organic_carbon": f"{oc:.2f}% ({oc_status})",
            "crop_suitability": suitability,
            "recommendations": recommendations[:3],
            "summary": f"Soil: {soil_type}, {n_status} N, {p_status} P, {k_status} K, {ph_status} pH. {suitability.title()} for {crop}."
        }
