import os
import sys
import argparse
import uvicorn

def setup():
    """Generate data and train models on first run."""
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    models_dir = os.path.join(os.path.dirname(__file__), "models", "saved_models")

    data_files = [
        os.path.join(data_dir, "crop_yield_data.csv"),
        os.path.join(data_dir, "weather_data.csv"),
        os.path.join(data_dir, "soil_data.csv"),
        os.path.join(data_dir, "historical_data.csv"),
    ]

    model_files = [
        os.path.join(models_dir, "xgb_model.pkl"),
        os.path.join(models_dir, "scaler.pkl"),
        os.path.join(models_dir, "label_encoders.pkl"),
        os.path.join(models_dir, "feature_cols.pkl"),
    ]

    missing_data = any(not os.path.exists(f) for f in data_files)
    missing_models = any(not os.path.exists(f) for f in model_files)

    if missing_data:
        print("📦 Fetching REAL data from HuggingFace + NASA POWER + ISRIC SoilGrids...")
        from data.real_data_pipeline import run
        run()

    if missing_models:
        print("🤖 Training ML models (XGBoost + LSTM)...")
        from models.train import train_all
        train_all()

    if not missing_data and not missing_models:
        print("✅ All data and models found. Skipping setup.")

def main():
    parser = argparse.ArgumentParser(description="YieldPredict AI - Crop Yield Forecasting & Insurance Risk Analyzer")
    parser.add_argument("--setup-only", action="store_true", help="Only generate data and train models")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    args = parser.parse_args()

    setup()

    if args.setup_only:
        print("Setup complete! Run without --setup-only to start the server.")
        return

    from backend.app import app
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse
    import pathlib

    frontend_dir = pathlib.Path(os.path.dirname(__file__)) / "frontend"

    app.mount("/static", StaticFiles(directory=str(frontend_dir / "static")), name="static")

    @app.get("/dashboard")
    async def dashboard():
        return FileResponse(str(frontend_dir / "templates" / "index.html"))

    @app.get("/dashboard/{rest:path}")
    async def dashboard_catchall():
        return FileResponse(str(frontend_dir / "templates" / "index.html"))

    print(f"\n" + "="*50)
    print("  🌾 YieldPredict AI - Server Starting")
    print(f"  📊 Dashboard: http://localhost:{args.port}/dashboard")
    print(f"  🔌 API:       http://localhost:{args.port}/api")
    print(f"  📋 Docs:      http://localhost:{args.port}/docs")
    print("="*50 + "\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

if __name__ == "__main__":
    main()
