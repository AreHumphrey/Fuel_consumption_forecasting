from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

TARGET = "fuel_consumption_l100km"

FEATURES_NUMERIC = [
    "engine_volume_l",
    "engine_power_hp",
    "vehicle_weight_kg",
    "vehicle_age_years",
    "avg_weekly_trips",
    "night_trip_ratio",
    "avg_speed_kmh",
    "fuel_price_rub_l",
]

FEATURES_CATEGORICAL = [
    "vehicle_type",
    "fuel_type",
    "transmission",
    "climate_zone",
    "road_quality",
    "season",
    "region_code",
]

CATBOOST_PARAMS = {
    "iterations": 1000,
    "learning_rate": 0.03,
    "depth": 6,
    "loss_function": "RMSE",
    "eval_metric": "MAE",
    "random_seed": 42,
    "verbose": 100,
    "early_stopping_rounds": 50,
    "cat_features": FEATURES_CATEGORICAL,
}

SEASON_COEFF = {
    "winter": 1.18,
    "summer": 1.0,
    "spring": 1.05,
    "autumn": 1.08,
}

REGION_COEFF = {
    "77": 1.0,
    "78": 1.02,
    "152": 1.15,
    "87": 1.25,
}