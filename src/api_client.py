import pandas as pd
from src.prices_loader import load_prices, get_price


class FuelPriceClient:
    
    STATIC_PRICES = {
        "ai92": 64.54,
        "ai95": 68.80,
        "ai98": 87.88,
        "diesel": 78.05,
        "gas": 31.15,
    }
    
    REGION_COEFFS = {
        "77": 1.00,
        "78": 1.02,
        "50": 1.01,
        "66": 0.98,
        "152": 1.15,
        "87": 1.25,
    }
    
    def get_average_prices_rf(self) -> pd.DataFrame:
        return load_prices()
    
    def get_prices_by_region(self, region_code: str = "77") -> pd.DataFrame:
        df = self.get_average_prices_rf()
        coeff = self.REGION_COEFFS.get(region_code, 1.05)
        df["price_rub_l"] = (df["price_rub_l"] * coeff).round(2)
        df["region_code"] = region_code
        return df
    
    def get_price(self, region_code: str, fuel_type: str) -> float:
        return get_price(region_code, fuel_type)