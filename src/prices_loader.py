import json
import pandas as pd
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).parent.parent
PRICES_FILE = BASE_DIR / "data" / "fuel_prices_ru.json"


def load_prices(region_code: Optional[str] = None) -> pd.DataFrame:
    with open(PRICES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    records = []
    for code, info in data["regions"].items():
        for fuel_type, price in [("ai92", info["ai92"]), ("ai95", info["ai95"]), ("diesel", info["diesel"])]:
            records.append({
                "region_code": code,
                "region_name": info["name"],
                "fuel_type": fuel_type,
                "price_rub_l": price
            })
    
    df = pd.DataFrame(records)
    
    if region_code:
        df = df[df["region_code"] == region_code]
    
    return df


def get_price(region_code: str, fuel_type: str) -> float:
    df = load_prices(region_code)
    row = df[df["fuel_type"] == fuel_type]
    if not row.empty:
        return row["price_rub_l"].values[0]
    return load_prices()["price_rub_l"].mean()