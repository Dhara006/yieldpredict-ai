import numpy as np
import pandas as pd
import pickle
import os
import json
import re

SAVED_MODELS_DIR = os.path.join(os.path.dirname(__file__), "saved_models")

def slug(name):
    return re.sub(r'[^a-zA-Z0-9_]', '_', name.replace(' ', '_').replace('(', '').replace(')', ''))

class YieldPredictor:
    def __init__(self):
        self.xgb_model = None
        self.bayes_model = None
        self.lstm_model = None
        self.scaler = None
        self.label_encoders = None
        self.feature_cols = None
        self.use_lstm = False
        self.crop_models = {}
        self.crop_model_map = {}
        self.load_models()

    def load_models(self):
        xgb_path = os.path.join(SAVED_MODELS_DIR, "xgb_model.pkl")
        bayes_path = os.path.join(SAVED_MODELS_DIR, "bayes_model.pkl")
        scaler_path = os.path.join(SAVED_MODELS_DIR, "scaler.pkl")
        le_path = os.path.join(SAVED_MODELS_DIR, "label_encoders.pkl")
        fc_path = os.path.join(SAVED_MODELS_DIR, "feature_cols.pkl")
        lstm_flag = os.path.join(SAVED_MODELS_DIR, "lstm_trained.flag")
        crop_map_path = os.path.join(SAVED_MODELS_DIR, "crop_model_map.json")

        with open(xgb_path, "rb") as f:
            self.xgb_model = pickle.load(f)
        if os.path.exists(bayes_path):
            with open(bayes_path, "rb") as f:
                self.bayes_model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        with open(le_path, "rb") as f:
            self.label_encoders = pickle.load(f)
        with open(fc_path, "rb") as f:
            self.feature_cols = pickle.load(f)

        if os.path.exists(lstm_flag):
            try:
                from tensorflow.keras.models import load_model
                lstm_path = os.path.join(SAVED_MODELS_DIR, "lstm_model.h5")
                self.lstm_model = load_model(lstm_path, compile=False)
                self.lstm_model.compile(optimizer="adam", loss="mse")
                self.use_lstm = True
            except (ImportError, OSError):
                pass

        if os.path.exists(crop_map_path):
            with open(crop_map_path) as f:
                self.crop_model_map = json.load(f)
            for crop_name in self.crop_model_map:
                fname = os.path.join(SAVED_MODELS_DIR, f"model_{slug(crop_name)}.pkl")
                if os.path.exists(fname):
                    with open(fname, "rb") as f:
                        self.crop_models[crop_name] = pickle.load(f)

    def _get_crop_model(self, crop):
        if crop in self.crop_models:
            return self.crop_models[crop]
        return None

    def _encode_features(self, input_data):
        df = pd.DataFrame([input_data])
        for col, le in self.label_encoders.items():
            if col in df.columns:
                val = str(df[col].iloc[0])
                if val in le.classes_:
                    df[col + "_enc"] = le.transform([val])[0]
                else:
                    df[col + "_enc"] = -1
            elif col + "_enc" in self.feature_cols:
                df[col + "_enc"] = -1

        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0

        if "avg_yield_5yr" in self.feature_cols and "avg_yield_5yr" not in df.columns:
            df["avg_yield_5yr"] = input_data.get("avg_yield_5yr", 0)

        X = df[self.feature_cols].fillna(0).values
        X_scaled = self.scaler.transform(X)
        return X_scaled

    def predict(self, input_data):
        crop = input_data.get("crop", "")
        crop_model = self._get_crop_model(crop)

        if crop_model is not None:
            scaler = crop_model["scaler"]
            features = crop_model["features"]
            le_dict = crop_model.get("label_encoders", {})
            model = crop_model["model"]

            df = pd.DataFrame([input_data])
            for col, le in le_dict.items():
                if col in df.columns:
                    val = str(df[col].iloc[0])
                    if val in le.classes_:
                        df[col + "_enc"] = le.transform([val])[0]
                    else:
                        df[col + "_enc"] = -1
                elif col + "_enc" in features:
                    df[col + "_enc"] = -1

            for col in features:
                if col not in df.columns:
                    df[col] = 0
            if "avg_yield_5yr" in features and "avg_yield_5yr" not in df.columns:
                df["avg_yield_5yr"] = input_data.get("avg_yield_5yr", 0)

            X = df[features].fillna(0).values
            X_scaled = scaler.transform(X)
            pred = model.predict(X_scaled)[0]
            return float(max(0, pred))

        X = self._encode_features(input_data)
        xgb_pred = self.xgb_model.predict(X)[0]

        lstm_pred = xgb_pred
        if self.use_lstm and self.lstm_model is not None:
            X_3d = X.reshape((1, 1, X.shape[1]))
            lstm_pred = self.lstm_model.predict(X_3d, verbose=0)[0][0]

        ensemble_pred = 0.6 * xgb_pred + 0.4 * lstm_pred
        return float(max(0, ensemble_pred))

    def predict_stage(self, input_data, days_after_sowing):
        base_pred = self.predict(input_data)
        stage_factors = {30: 0.3, 60: 0.6, 90: 0.85, 120: 0.95, 150: 1.0}
        factor = stage_factors.get(days_after_sowing, 1.0)
        return max(0, base_pred * factor)

    def predict_with_uncertainty(self, input_data, n_simulations=1000):
        X = self._encode_features(input_data)
        base_pred = float(self.xgb_model.predict(X)[0])
        base_pred = max(0.01, base_pred)

        if self.bayes_model is not None:
            _, bayes_std = self.bayes_model.predict(X, return_std=True)
            uncertainty_std = float(bayes_std[0])
        else:
            uncertainty_std = 0.1 * base_pred

        cv = min(2.0, uncertainty_std / base_pred)
        sigma = cv
        mu = np.log(base_pred) - 0.5 * sigma ** 2

        from scipy.stats import norm
        mean_val = float(np.exp(mu + sigma ** 2 / 2))
        median_val = float(np.exp(mu))
        var_log = (np.exp(sigma ** 2) - 1) * np.exp(2 * mu + sigma ** 2)
        std_val = float(np.sqrt(var_log)) if var_log > 0 else 0.0
        p10 = float(np.exp(mu + sigma * norm.ppf(0.10)))
        p90 = float(np.exp(mu + sigma * norm.ppf(0.90)))

        return {
            "mean": mean_val,
            "median": median_val,
            "std": std_val,
            "p10": p10,
            "p90": p90
        }

    def get_model_info(self, crop):
        info = self.crop_model_map.get(crop, {})
        if info:
            return {"model": info.get("best_model_name"), "r2": info.get("best_r2"), "mae": info.get("best_mae")}
        return {"model": "XGBoost (global)", "r2": 0.956, "mae": 0.659}

predictor = YieldPredictor()
