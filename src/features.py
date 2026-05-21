import pandas as pd
import numpy as np
from datetime import datetime
from src.prices_loader import get_price
from src.config import SEASON_COEFF, REGION_COEFF, TARGET


def add_seasonal_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    df = df.copy()
    if date_col in df:
        dates = pd.to_datetime(df[date_col])
        df["month"] = dates.dt.month
        df["season"] = dates.dt.month.map({
            12: "winter", 1: "winter", 2: "winter",
            3: "spring", 4: "spring", 5: "spring",
            6: "summer", 7: "summer", 8: "summer",
            9: "autumn", 10: "autumn", 11: "autumn"
        })
    else:
        df["season"] = "summer"
    
    df["season_coeff"] = df["season"].map(SEASON_COEFF)
    return df


def add_regional_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    climate_map = {
        "77": "moderate", "78": "moderate", "52": "moderate",
        "66": "continental", "75": "continental",
        "14": "northern", "87": "northern", "49": "northern",
    }
    df["climate_zone"] = df["region_code"].astype(str).map(climate_map).fillna("moderate")
    
    road_map = {
        "77": "asphalt", "78": "asphalt", "50": "asphalt",
        "14": "mixed", "87": "gravel"
    }
    df["road_quality"] = df["region_code"].astype(str).map(road_map).fillna("asphalt")
    
    df["region_coeff"] = df["region_code"].astype(str).map(
        lambda x: REGION_COEFF.get(x, 1.0)
    )
    return df


def add_fuel_prices(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    def get_row_price(row):
        return get_price(str(row["region_code"]), row["fuel_type"])
    
    df["fuel_price_rub_l"] = df.apply(get_row_price, axis=1)
    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    if TARGET not in df.columns:
        df[TARGET] = (
            4.5 +
            1.2 * df["engine_volume_l"] +
            0.0015 * df["vehicle_weight_kg"] +
            0.08 * df["vehicle_age_years"] +
            0.4 * (df["avg_speed_kmh"] < 35).astype(int) +
            0.3 * df["night_trip_ratio"] +
            np.random.normal(0, 0.4, size=len(df))
        )
        df[TARGET] *= df.get("season_coeff", 1.0)
        df[TARGET] = df[TARGET].clip(3.5, 25).round(2)
    
    if "cost_per_100km_rub" not in df.columns:
        df["cost_per_100km_rub"] = (df[TARGET] * df["fuel_price_rub_l"]).round(2)
    
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_seasonal_features(df)
    df = add_regional_features(df)
    df = add_fuel_prices(df)
    df = create_target(df)
    return df