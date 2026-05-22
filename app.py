from flask import Flask, request, jsonify, render_template
from src.model import FuelConsumptionModel
from src.api_client import FuelPriceClient
from src.features import prepare_features
from src.prices_loader import get_price
import pandas as pd
import os

app = Flask(__name__)

model = None
fuel_client = None


def init_app():
    global model, fuel_client
    model_path = os.getenv("MODEL_PATH", "models/catboost_ru_v1.cbm")
    model = FuelConsumptionModel.load(model_path)
    fuel_client = FuelPriceClient()


@app.route("/")
def index():
    return render_template("index.html")


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
    
    region = data.get("region_code", "77")
    fuel_type = data.get("fuel_type", "ai95")
    
    price = get_price(region, fuel_type)
    
    return jsonify({
        "predicted_consumption_l100km": round(prediction, 2),
        "fuel_price_rub_l": round(price, 2),
        "cost_per_100km_rub": round(prediction * price, 2)
    }), 200


@app.route("/prices", methods=["GET"])
def get_prices():
    region = request.args.get("region_code", "77")
    if not fuel_client:
        return jsonify({"error": "API client not initialized"}), 503
    prices = fuel_client.get_prices_by_region(region)
    return jsonify(prices.to_dict(orient="records")), 200


if __name__ == "__main__":
    init_app()
    app.run(host="0.0.0.0", port=5000, debug=True)