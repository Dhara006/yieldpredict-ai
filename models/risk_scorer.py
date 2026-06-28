import numpy as np
from scipy import stats

class BayesianRiskScorer:
    def __init__(self):
        self.crop_priors = {
            "Rice": {"alpha": 4, "beta": 6},
            "Wheat": {"alpha": 3, "beta": 7},
            "Maize": {"alpha": 5, "beta": 5},
            "Cotton": {"alpha": 6, "beta": 4},
            "Sugarcane": {"alpha": 3.5, "beta": 6.5},
            "Groundnut": {"alpha": 5.5, "beta": 4.5},
            "Soybean": {"alpha": 4.5, "beta": 5.5}
        }
        self.drought_priors = {
            "none": {"alpha": 1, "beta": 9},
            "mild": {"alpha": 2.5, "beta": 7.5},
            "moderate": {"alpha": 4.5, "beta": 5.5},
            "severe": {"alpha": 7, "beta": 3}
        }
        self.pest_priors = {
            "low": {"alpha": 1, "beta": 9},
            "medium": {"alpha": 3, "beta": 7},
            "high": {"alpha": 5, "beta": 5}
        }

    def _posterior_mean(self, prior, successes=0, failures=0):
        alpha_post = prior["alpha"] + failures
        beta_post = prior["beta"] + successes
        return alpha_post / (alpha_post + beta_post)

    def _beta_variance(self, prior, successes=0, failures=0):
        a = prior["alpha"] + failures
        b = prior["beta"] + successes
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    def compute_risk_score(self, input_data, yield_prediction, uncertainty_std):
        crop_risk = self._posterior_mean(
            self.crop_priors.get(input_data.get("crop", "Wheat"), {"alpha": 4, "beta": 6})
        )
        drought_risk = self._posterior_mean(
            self.drought_priors.get(input_data.get("drought_history", "none"), {"alpha": 1, "beta": 9})
        )
        pest_risk = self._posterior_mean(
            self.pest_priors.get(input_data.get("pest_history", "low"), {"alpha": 1, "beta": 9})
        )

        irrigation_pct = input_data.get("irrigation_pct", 50)
        irrig_prior = {"alpha": 1 + (100 - irrigation_pct) / 10, "beta": 1 + irrigation_pct / 10}
        irrigation_risk = self._posterior_mean(irrig_prior)

        avg_5yr = input_data.get("avg_yield_5yr", 0)
        if avg_5yr > 0 and yield_prediction:
            deviation = (yield_prediction - avg_5yr) / avg_5yr
            if deviation < 0:
                failures = int(min(10, abs(deviation) * 20))
                hist_risk = self._posterior_mean({"alpha": 2, "beta": 8}, failures=failures)
            else:
                hist_risk = self._posterior_mean({"alpha": 2, "beta": 8}, successes=1)
        else:
            hist_risk = self._posterior_mean({"alpha": 2, "beta": 8})

        weather_failures = int(min(10, uncertainty_std / (yield_prediction + 0.01) * 20))
        weather_risk = self._posterior_mean({"alpha": 2, "beta": 8}, failures=weather_failures)

        raw_risk = (
            0.25 * crop_risk +
            0.20 * drought_risk +
            0.15 * pest_risk +
            0.10 * irrigation_risk +
            0.15 * hist_risk +
            0.15 * weather_risk
        )

        risk_score = min(10, max(1, round(raw_risk * 10, 1)))

        crop_var = self._beta_variance(
            self.crop_priors.get(input_data.get("crop", "Wheat"), {"alpha": 4, "beta": 6})
        )
        drought_var = self._beta_variance(
            self.drought_priors.get(input_data.get("drought_history", "none"), {"alpha": 1, "beta": 9})
        )
        pest_var = self._beta_variance(
            self.pest_priors.get(input_data.get("pest_history", "low"), {"alpha": 1, "beta": 9})
        )
        irrig_var = self._beta_variance(irrig_prior)
        if avg_5yr > 0 and yield_prediction:
            deviation = (yield_prediction - avg_5yr) / avg_5yr
            if deviation < 0:
                f = int(min(10, abs(deviation) * 20))
                hist_var = self._beta_variance({"alpha": 2, "beta": 8}, failures=f)
            else:
                hist_var = self._beta_variance({"alpha": 2, "beta": 8}, successes=1)
        else:
            hist_var = self._beta_variance({"alpha": 2, "beta": 8})
        weather_var = self._beta_variance({"alpha": 2, "beta": 8}, failures=weather_failures)

        total_var = (
            0.25 ** 2 * crop_var + 0.20 ** 2 * drought_var + 0.15 ** 2 * pest_var +
            0.10 ** 2 * irrig_var + 0.15 ** 2 * hist_var + 0.15 ** 2 * weather_var
        )
        total_std = np.sqrt(total_var)
        ci_width = 2 * 1.645 * total_std
        confidence = max(0, min(100, 100 - ci_width * 100))

        return {"risk_score": risk_score, "confidence": round(confidence, 1)}

    def monte_carlo_risk(self, mean_yield, std_yield, n_simulations=10000):
        cv = std_yield / (mean_yield + 0.01)
        sigma = min(2.0, cv)
        mean_yield = max(0.01, mean_yield)
        mu = np.log(mean_yield) - 0.5 * sigma ** 2
        threshold = mean_yield * 0.7

        from scipy.stats import norm, lognorm
        default_prob = float(lognorm.cdf(threshold, s=sigma, scale=np.exp(mu)))

        if sigma > 1e-6:
            z = norm.ppf(default_prob)
            e_term = np.exp(mu + sigma ** 2 / 2) * norm.cdf(z - sigma)
            expected_loss = max(0.0, float(threshold * default_prob - e_term))

            z05 = norm.ppf(0.05)
            var_95 = float(mean_yield - np.exp(mu + sigma * z05))
            cvar_95 = float(mean_yield - (np.exp(mu + sigma ** 2 / 2) * norm.cdf(z05 - sigma) / 0.05))
        else:
            expected_loss = 0.0
            var_95 = 0.0
            cvar_95 = 0.0

        return {
            "default_probability": round(default_prob, 4),
            "expected_loss": round(expected_loss, 4),
            "value_at_risk_95": round(max(0, var_95), 4),
            "conditional_var_95": round(max(0, cvar_95), 4),
            "coefficient_of_variation": round(float(cv), 4)
        }

risk_scorer = BayesianRiskScorer()
