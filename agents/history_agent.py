import pandas as pd
import os
import numpy as np
from .base_agent import BaseAgent

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

class HistoryAgent(BaseAgent):
    def __init__(self):
        super().__init__("History Agent")
        self.hist_data = pd.read_csv(os.path.join(DATA_DIR, "historical_data.csv"))

    def analyze(self, context):
        district = context.get("district", "")
        crop = context.get("crop", "Wheat")

        subset = self.hist_data[
            (self.hist_data["district"] == district) &
            (self.hist_data["crop"] == crop)
        ]
        if subset.empty:
            subset = self.hist_data[self.hist_data["crop"] == crop]
        if subset.empty:
            subset = self.hist_data

        avg_yield = subset["avg_yield_5yr"].mean()
        max_yield = subset["avg_yield_5yr"].max()
        min_yield = subset["avg_yield_5yr"].min()
        yield_trend = np.polyfit(range(len(subset)), subset["avg_yield_5yr"], 1)[0] if len(subset) > 1 else 0
        pest = subset["pest_history"].value_counts().idxmax()
        drought = subset["drought_history"].value_counts().idxmax()
        area = subset["area_trend"].value_counts().idxmax()

        latest = subset.loc[subset["year"].idxmax()] if "year" in subset.columns and not subset.empty else None
        if latest is not None:
            last_5 = [
                latest.get("yield_t_1", avg_yield),
                latest.get("yield_t_2", avg_yield),
                latest.get("yield_t_3", avg_yield),
                latest.get("yield_t_4", avg_yield),
                latest.get("yield_t_5", avg_yield),
            ]
        else:
            last_5 = [avg_yield] * 5

        return {
            "avg_yield_5yr": round(float(avg_yield), 2),
            "max_yield_5yr": round(float(max_yield), 2),
            "min_yield_5yr": round(float(min_yield), 2),
            "yield_trend": "increasing" if yield_trend > 0.01 else "declining" if yield_trend < -0.01 else "stable",
            "trend_strength": round(float(yield_trend), 4),
            "pest_history": pest,
            "drought_history": drought,
            "area_trend": area,
            "last_5_yields": [round(float(y), 2) for y in last_5],
            "summary": f"5yr avg: {avg_yield:.2f} t/ha ({max_yield:.2f} max, {min_yield:.2f} min). "
                       f"Trend: {'rising' if yield_trend > 0 else 'falling'}. Pest: {pest}, Drought: {drought}."
        }
