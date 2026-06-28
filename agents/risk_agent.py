from .base_agent import BaseAgent
from models.predict import predictor
from models.risk_scorer import risk_scorer
import numpy as np

class RiskAgent(BaseAgent):
    def __init__(self):
        super().__init__("Risk Agent")

    def analyze(self, context):
        district = context.get("district", "")
        crop = context.get("crop", "Wheat")

        pred_result = predictor.predict_with_uncertainty(context)
        yield_mean = pred_result["mean"]
        yield_std = pred_result["std"]

        risk_result = risk_scorer.compute_risk_score(context, yield_mean, yield_std)
        mc_result = risk_scorer.monte_carlo_risk(yield_mean, yield_std)
        climate_impact = self._simulate_climate_impact(context, yield_mean)
        premium = self._calculate_premium(risk_result["risk_score"], mc_result["default_probability"])

        return {
            "yield_forecast": {
                "mean_t_per_ha": round(yield_mean, 2),
                "std_t_per_ha": round(yield_std, 2),
                "p10": round(pred_result["p10"], 2),
                "p90": round(pred_result["p90"], 2)
            },
            "risk_score": risk_result["risk_score"],
            "confidence": risk_result["confidence"],
            "monte_carlo": mc_result,
            "climate_impact": climate_impact,
            "premium_estimate": premium,
            "summary": f"Risk: {risk_result['risk_score']}/10 ({risk_result['confidence']}%), "
                       f"yield {yield_mean:.2f}±{yield_std:.2f} t/ha, "
                       f"default prob: {mc_result['default_probability']:.1%}"
        }

    def _simulate_climate_impact(self, context, base_yield):
        scenarios = {}
        for ti in [1.0, 2.0, 3.0]:
            for ri in [-0.3, -0.1, 0.1]:
                impact = -0.05 * ti + 0.2 * ri
                scenarios[f"+{ti}°C, {ri*100:+.0f}% rain"] = round(float(base_yield * (1 + impact)), 2)
        return scenarios

    def _calculate_premium(self, risk_score, default_prob):
        rate = 0.02 + default_prob * 0.5
        if risk_score > 3: rate = 0.05 + default_prob * 0.8
        if risk_score > 6: rate = 0.10 + default_prob * 1.2
        rate = min(0.25, rate)
        return {
            "premium_rate_pct": round(rate * 100, 2),
            "premium_per_ha_rupees": round(rate * 50000, 0),
            "coverage_amount_rupees": 50000
        }
