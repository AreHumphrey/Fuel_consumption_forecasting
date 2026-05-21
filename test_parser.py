from src.api_client import FuelPriceClient

client = FuelPriceClient()
prices = client.get_average_prices_rf()
print(prices)