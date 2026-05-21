import argparse
import pandas as pd
import logging
from src.model import FuelConsumptionModel
from src.features import prepare_features

logging.basicConfig(level=logging.INFO, format="%(message)s")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Путь к модели .cbm")
    parser.add_argument("--input", required=True, help="Путь к входным данным CSV")
    parser.add_argument("--output", required=True, help="Путь для сохранения прогнозов")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    model = FuelConsumptionModel.load(args.model)
    df = pd.read_csv(args.input)
    
    df = prepare_features(df)
    features = model.feature_names
    X = df[features]
    
    predictions = model.predict(X)
    df["predicted_consumption"] = predictions.round(2)
    
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    logging.info(f"Прогнозы сохранены: {args.output}")