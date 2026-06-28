import pandas as pd
import os
from .base_agent import BaseAgent

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

class WeatherAgent(BaseAgent):
    def __init__(self):
        super().__init__("Weather Agent")
        self.weather_data = pd.read_csv(os.path.join(DATA_DIR, "weather_data.csv"))

    def analyze(self, context):
        district = context.get("district", "")
        month = int(context.get("month", 6))

        district_data = self.weather_data[self.weather_data["district"] == district]
        if district_data.empty:
            district_data = self.weather_data

        month_data = district_data[district_data["month"] == month]
        if month_data.empty:
            month_data = district_data

        actual_temp = month_data["avg_temp_c"].mean()
        actual_rainfall = month_data["rainfall_mm"].mean()
        actual_humidity = month_data["humidity_pct"].mean()
        actual_sunshine = month_data["sunshine_hours"].mean()
        actual_wind = month_data["wind_speed_kmh"].mean()
        temp_min = month_data["avg_temp_c"].min()
        temp_max = month_data["avg_temp_c"].max()

        latest = month_data[month_data["year"] == month_data["year"].max()]
        if not latest.empty:
            latest_temp = latest["avg_temp_c"].iloc[0]
            latest_rainfall = latest["rainfall_mm"].iloc[0]
        else:
            latest_temp = actual_temp
            latest_rainfall = actual_rainfall

        if month in [6, 7, 8, 9]:
            monsoon = "active monsoon"
        elif month in [10, 11]:
            monsoon = "retreating monsoon"
        else:
            monsoon = "dry season"

        return {
            "predicted_temp_c": round(float(latest_temp), 1),
            "predicted_rainfall_mm": round(float(max(0, latest_rainfall)), 1),
            "avg_temp_c": round(float(actual_temp), 1),
            "rainfall_mm": round(float(max(0, actual_rainfall)), 1),
            "temp_min_c": round(float(temp_min), 1),
            "temp_max_c": round(float(temp_max), 1),
            "humidity_pct": round(float(actual_humidity), 1),
            "sunshine_hours": round(float(actual_sunshine), 1),
            "wind_speed_kmh": round(float(actual_wind), 1),
            "monsoon_status": monsoon,
            "years_of_data": int(month_data["year"].nunique()),
            "heat_stress_risk": "high" if latest_temp > 35 else "moderate" if latest_temp > 30 else "low",
            "drought_risk": "high" if latest_rainfall < 10 else "moderate" if latest_rainfall < 40 else "low",
            "summary": f"Weather for {district}: {latest_temp:.1f}°C, {max(0, latest_rainfall):.1f}mm rain, {monsoon}"
        }
