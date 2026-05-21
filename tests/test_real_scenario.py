import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.model import FuelConsumptionModel
from src.features import prepare_features
from src.prices_loader import get_price


def run_real_scenario():

    input_data = [{
        "engine_volume_l": 2.5,
        "engine_power_hp": 200,
        "vehicle_weight_kg": 1620,
        "vehicle_age_years": 3,
        "avg_weekly_trips": 12,
        "night_trip_ratio": 0.15,
        "avg_speed_kmh": 32,         
        "vehicle_type": "sedan",
        "fuel_type": "ai95",
        "transmission": "automatic",
        "region_code": "77",        
        "season": "winter"           
    }]
    df = pd.DataFrame(input_data)

    prepared_df = prepare_features(df)

    model_path = "models/catboost_ru_v1.cbm"
    if not Path(model_path).exists():
        return

    model = FuelConsumptionModel.load(model_path)

    features = model.feature_names
    X = prepared_df[features]
    predicted_consumption = model.predict(X)[0]

    fuel_price = get_price("77", "ai95")
    trip_distance_km = 450 

    cost_per_100km = predicted_consumption * fuel_price

    liters_needed = (predicted_consumption / 100) * trip_distance_km

    total_trip_cost = liters_needed * fuel_price

    print("\n" + "=" * 60)

    print("=" * 60)
    print(f"регион:         Москва (код 77)")
    print(f"автомобиль:     Седан 2.5L, АКПП, 3 года эксплуатации")
    print(f"условия:        Зима, городская скорость ~32 км/ч")
    print("-" * 60)
    print(f"цена АИ-95:     {fuel_price:.2f} руб/л")
    print(f"прогноз расхода:{predicted_consumption:.2f} л/100 км")
    print(f"стоимость 100 км: {cost_per_100km:.2f} руб.")
    print("-" * 60)
    print(f"дистанция:       {trip_distance_km} км")
    print(f"необходимо бензина:   {liters_needed:.2f} л.")
    print(f"итого {total_trip_cost:.2f} руб.")
    print("=" * 60)




if __name__ == "__main__":
    run_real_scenario()