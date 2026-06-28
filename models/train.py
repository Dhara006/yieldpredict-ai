import numpy as np
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import BayesianRidge
import xgboost as xgb

SAVED_MODELS_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

def load_and_merge_data():
    yield_df = pd.read_csv(os.path.join(DATA_DIR, "crop_yield_data.csv"))
    weather_df = pd.read_csv(os.path.join(DATA_DIR, "weather_data.csv"))
    soil_df = pd.read_csv(os.path.join(DATA_DIR, "soil_data.csv"))
    hist_df = pd.read_csv(os.path.join(DATA_DIR, "historical_data.csv"))

    merged = yield_df.merge(weather_df, on=["district", "year", "month"], how="left")
    merged = merged.merge(soil_df, on=["district"], how="left")

    hist_agg = hist_df.groupby(["district", "crop"]).agg({
        "avg_yield_5yr": "mean",
        "pest_history": lambda x: x.mode().iloc[0] if not x.mode().empty else "low",
        "drought_history": lambda x: x.mode().iloc[0] if not x.mode().empty else "none"
    }).reset_index()

    merged = merged.merge(hist_agg, on=["district", "crop"], how="left")
    return merged

def preprocess(df):
    le_dict = {}

    for col in ["district", "crop", "soil_type", "pest_history", "drought_history", "area_trend"]:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = df[col].astype(str)
            df[col + "_enc"] = le.fit_transform(df[col])
            le_dict[col] = le

    feature_cols = [
        "year", "month", "sowing_area_ha", "irrigation_pct", "fertilizer_kg_per_ha",
        "pesticide_kg_per_ha", "avg_temp_c", "rainfall_mm", "humidity_pct",
        "sunshine_hours", "wind_speed_kmh", "N_mg_per_kg", "P_mg_per_kg",
        "K_mg_per_kg", "pH", "organic_carbon_pct", "zinc_mg_per_kg",
        "iron_mg_per_kg", "copper_mg_per_kg", "manganese_mg_per_kg",
        "boron_mg_per_kg", "avg_yield_5yr"
    ] + [col + "_enc" for col in ["district", "crop", "soil_type", "pest_history", "drought_history"]]

    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    available = [c for c in feature_cols if c in df.columns]
    X = df[available]
    y = df["yield_per_ha"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, le_dict, available

def train_xgboost(X_train, y_train, X_test, y_test):
    model = xgb.XGBRegressor(
        n_estimators=200, max_depth=7, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"XGBoost - MAE: {mae:.3f}, R2: {r2:.3f}")
    return model

def train_bayesian_ridge(X_train, y_train, X_test, y_test):
    model = BayesianRidge(
        alpha_1=1e-6, alpha_2=1e-6,
        lambda_1=1e-6, lambda_2=1e-6,
        compute_score=True
    )
    model.fit(X_train, y_train)
    y_pred, y_std = model.predict(X_test, return_std=True)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"BayesianRidge - MAE: {mae:.3f}, R2: {r2:.3f}, Mean STD: {np.mean(y_std):.3f}")
    return model

def train_lstm(X_train, y_train, X_test, y_test):
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping

        X_train_3d = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test_3d = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

        model = Sequential([
            LSTM(64, input_shape=(1, X_train.shape[1]), return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        es = EarlyStopping(patience=10, restore_best_weights=True)

        model.fit(X_train_3d, y_train, epochs=50, batch_size=32,
                  validation_data=(X_test_3d, y_test), callbacks=[es], verbose=0)

        y_pred = model.predict(X_test_3d, verbose=0).flatten()
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"LSTM - MAE: {mae:.3f}, R2: {r2:.3f}")
        return model
    except ImportError:
        print("TensorFlow not available, skipping LSTM training")
        return None

def train_all():
    print("Loading and merging data...")
    df = load_and_merge_data()
    print(f"Dataset shape: {df.shape}")

    X, y, scaler, le_dict, feature_cols = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\nTraining XGBoost...")
    xgb_model = train_xgboost(X_train, y_train, X_test, y_test)

    print("\nTraining BayesianRidge...")
    bayes_model = train_bayesian_ridge(X_train, y_train, X_test, y_test)

    print("\nTraining LSTM...")
    lstm_model = train_lstm(X_train, y_train, X_test, y_test)

    print("\nSaving models and artifacts...")
    with open(os.path.join(SAVED_MODELS_DIR, "xgb_model.pkl"), "wb") as f:
        pickle.dump(xgb_model, f)

    with open(os.path.join(SAVED_MODELS_DIR, "bayes_model.pkl"), "wb") as f:
        pickle.dump(bayes_model, f)

    if lstm_model:
        lstm_model.save(os.path.join(SAVED_MODELS_DIR, "lstm_model.h5"))
        with open(os.path.join(SAVED_MODELS_DIR, "lstm_trained.flag"), "w") as f:
            f.write("1")

    with open(os.path.join(SAVED_MODELS_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    with open(os.path.join(SAVED_MODELS_DIR, "label_encoders.pkl"), "wb") as f:
        pickle.dump(le_dict, f)

    with open(os.path.join(SAVED_MODELS_DIR, "feature_cols.pkl"), "wb") as f:
        pickle.dump(feature_cols, f)

    print("Training complete! Models saved to", SAVED_MODELS_DIR)

if __name__ == "__main__":
    train_all()
