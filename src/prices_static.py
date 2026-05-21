import pandas as pd

STATIC_PRICES = {
    "ai92": 64.54,
    "ai95": 68.80,
    "ai98": 87.88,
    "diesel": 78.05,
    "gas": 31.15
}

def get_static_prices() -> pd.DataFrame:
    return pd.DataFrame(
        list(STATIC_PRICES.items()), 
        columns=["fuel_type", "price_rub_l"]
    )