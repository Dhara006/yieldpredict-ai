class YieldExplainer:
    def __init__(self):
        self.templates = {
            "good_yield": [
                "Good news, kisaan! Your {crop} crop is looking strong this season. "
                "We expect about {yield_val} tonnes per hectare. This is {comparison} than your last 5-year average of {avg} t/ha. "
                "The weather has been cooperative with good rainfall and suitable temperatures. "
                "Keep up the good work with your current practices!",

                "Bohut achha! Your {crop} field is thriving. Based on our analysis, "
                "you should get around {yield_val} tonnes per hectare - that's {comparison} than previous years. "
                "The soil is healthy and monsoon is on track. Happy farming!"
            ],
            "average_yield": [
                "Your {crop} crop is expected to yield about {yield_val} tonnes per hectare, "
                "which is {comparison} than your 5-year average of {avg} t/ha. "
                "Consider applying recommended fertilizers and monitoring pest activity. "
                "The weather outlook is mixed - stay prepared for variable rainfall.",

                "Bhai, your {crop} yield prediction is around {yield_val} t/ha. "
                "Compared to last 5 years ({avg} t/ha), this is {comparison}. "
                "Keep an eye on soil moisture and pest control for best results."
            ],
            "poor_yield": [
                "Your {crop} crop may face some challenges this season. "
                "Predicted yield is {yield_val} tonnes per hectare, which is {comparison} than your average of {avg} t/ha. "
                "Weather patterns are less favorable and soil nutrients need attention. "
                "Consider crop insurance to protect against potential losses.",

                "Kisaan bhai, is baar {crop} ki paidaawar kam ho sakti hai - around {yield_val} t/ha. "
                "Pichle 5 saal ke average ({avg} t/ha) se {comparison} hai. "
                "Mausam aur mitti dono mein sudhaar ki zaroorat hai. "
                "Fasal beema jaroor karaayein."
            ]
        }

    def explain(self, context, yield_prediction, risk_info, weather_info, soil_info, history_info):
        avg_5yr = context.get("avg_yield_5yr", history_info.get("avg_yield_5yr", yield_prediction))
        diff_pct = ((yield_prediction - avg_5yr) / (avg_5yr + 0.01)) * 100
        crop = context.get("crop", "crop")

        if diff_pct > 10:
            category = "good_yield"
            comparison = f"{abs(diff_pct):.0f}% higher"
        elif diff_pct > -10:
            category = "average_yield"
            comparison = "similar" if abs(diff_pct) < 3 else f"{abs(diff_pct):.0f}% lower"
        else:
            category = "poor_yield"
            comparison = f"{abs(diff_pct):.0f}% lower"

        narrative = self.templates[category][0].format(
            crop=crop.capitalize(),
            yield_val=round(yield_prediction, 2),
            avg=round(avg_5yr, 2),
            comparison=comparison
        )

        bullet_points = [
            f"📊 Predicted yield: {yield_prediction:.2f} t/ha",
            f"📈 vs 5-year average: {comparison} ({avg_5yr:.2f} t/ha)",
            f"🌡️ Weather: {weather_info.get('predicted_temp_c', 'N/A')}°C, {weather_info.get('predicted_rainfall_mm', 'N/A')}mm rain",
            f"🌱 Soil: {soil_info.get('soil_type', 'N/A')} - {soil_info.get('crop_suitability', 'N/A')}",
            f"🔢 Risk Score: {risk_info.get('risk_score', 'N/A')}/10",
            f"💰 Insurance premium estimate: ₹{risk_info.get('premium_estimate', {}).get('premium_per_ha_rupees', 0):.0f}/ha"
        ]

        recommendations = []
        if soil_info.get("recommendations"):
            recommendations = soil_info.get("recommendations")[:2]
        if risk_info.get("risk_score", 0) > 5:
            recommendations.append("Consider purchasing crop insurance")
        if weather_info.get("drought_risk") == "high":
            recommendations.append("Prepare for irrigation/drought conditions")

        return {
            "narrative": narrative,
            "summary_bullets": bullet_points,
            "recommendations": recommendations,
            "category": category.replace("_", " ").title()
        }

explainer = YieldExplainer()
