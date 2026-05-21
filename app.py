from flask import Flask, request, jsonify
from src.model import FuelConsumptionModel
from src.api_client import OMTFuelClient
from src.features import prepare_features
import pandas as pd
import os

app = Flask(__name__)

model = None
fuel_client = None


def init_app():
    global model, fuel_client
    model_path = os.getenv("MODEL_PATH", "models/catboost_ru_v1.cbm")
    api_token = os.getenv("OMT_API_TOKEN")
    
    model = FuelConsumptionModel.load(model_path)
    fuel_client = OMTFuelClient(token=api_token)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    
    input_df = pd.DataFrame([data])
    input_df = prepare_features(input_df)
    
    features = model.feature_names
    X = input_df[features]
    
    prediction = model.predict(X)[0]
    
    region = data.get("region_code")
    fuel_type = data.get("fuel_type")
    
    cost_info = {}
    if fuel_client and region and fuel_type:
        prices = fuel_client.get_stations_prices(region_code=region, fuel_type=fuel_type)
        if prices is not None and not prices.empty:
            price = prices["price_rub_l"].values[0]
            cost_info = {
                "fuel_price_rub_l": round(price, 2),
                "cost_per_100km_rub": round(prediction * price, 2)
            }
    
    return jsonify({
        "predicted_consumption_l100km": round(prediction, 2),
        **cost_info
    }), 200


@app.route("/prices", methods=["GET"])
def get_prices():
    region = request.args.get("region_code")
    fuel_type = request.args.get("fuel_type")
    
    if not fuel_client:
        return jsonify({"error": "API client not initialized"}), 503
    
    prices = fuel_client.get_stations_prices(region_code=region, fuel_type=fuel_type)
    
    if prices is None:
        return jsonify({"error": "Failed to fetch prices"}), 500
    
    return jsonify(prices.to_dict(orient="records")), 200


if __name__ == "__main__":
    init_app()
    app.run(host="0.0.0.0", port=5000, debug=True)