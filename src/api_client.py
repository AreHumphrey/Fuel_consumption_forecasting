import requests
import pandas as pd
from bs4 import BeautifulSoup
from typing import Optional, Dict, List
import logging
import re

logger = logging.getLogger(__name__)


class FuelPriceClient:
    """
    Клиент для получения цен на топливо.
    Использует парсинг открытых источников (benzin-price.ru), 
    так как бесплатные прямые API отсутствуют.
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        })

    def get_average_prices_rf(self) -> Optional[pd.DataFrame]:
        """
        Получает средние цены на топливо по РФ.
        Возвращает DataFrame: [fuel_type, price_rub_l]
        """
        url = "https://benzin-price.ru/"
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'lxml')
            
            # Ищем таблицу или блоки с ценами. 
            # На benzin-price.ru структура может меняться, ищем по ключевым словам
            prices = {}
            
            # Примерный парсинг блоков с ценами (нужно адаптировать под верстку)
            # Обычно цены лежат в div с классами типа .price-item или в таблице
            # Для надежности попробуем найти текст "АИ-92", "АИ-95" и взять цену рядом
            
            text_content = soup.get_text()
            
            # Регулярные выражения для поиска цен (грубый метод, но рабочий для демо)
            # Ищем паттерны: "АИ-92 ... 64.50 руб"
            patterns = {
                "ai92": r"АИ-92[^\d]*(\d+[\.,]\d+)",
                "ai95": r"АИ-95[^\d]*(\d+[\.,]\d+)",
                "ai98": r"АИ-98[^\d]*(\d+[\.,]\d+)",
                "diesel": r"ДТ[^\d]*(\d+[\.,]\d+)"
            }
            
            for fuel_key, pattern in patterns.items():
                match = re.search(pattern, text_content)
                if match:
                    price_str = match.group(1).replace(',', '.')
                    prices[fuel_key] = float(price_str)
                    
            if not prices:
                logger.warning("Не удалось спарсить цены, используем заглушки")
                return self._get_fallback_prices()
                
            df = pd.DataFrame(list(prices.items()), columns=["fuel_type", "price_rub_l"])
            return df
            
        except Exception as e:
            logger.error(f"Ошибка парсинга цен: {e}")
            return self._get_fallback_prices()

    def get_prices_by_region(self, region_name: str = "Москва") -> Optional[pd.DataFrame]:
        """
        Пытается получить цены по конкретному региону.
        Для упрощения в демо-версии возвращает цены РФ с небольшим коэффициентом.
        """
        df_rf = self.get_average_prices_rf()
        if df_rf is None:
            return self._get_fallback_prices()
        
        # Эмуляция региональной наценки (в реальности нужен отдельный парсер по городам)
        coefficients = {
            "Москва": 1.0,
            "Санкт-Петербург": 1.02,
            "Владивосток": 1.15,
            "Новосибирск": 0.98,
            "Екатеринбург": 1.01
        }
        
        coeff = coefficients.get(region_name, 1.05) # Дефолт +5% для регионов
        df_rf["price_rub_l"] = df_rf["price_rub_l"] * coeff
        df_rf["region"] = region_name
        
        return df_rf

    def _get_fallback_prices(self) -> pd.DataFrame:
        """Возвращает жестко заданные средние цены, если парсинг не удался."""
        logger.info("Используются резервные данные о ценах (май 2026)")
        data = [
            {"fuel_type": "ai92", "price_rub_l": 64.54},
            {"fuel_type": "ai95", "price_rub_l": 68.80},
            {"fuel_type": "ai98", "price_rub_l": 87.88},
            {"fuel_type": "diesel", "price_rub_l": 78.05},
        ]
        return pd.DataFrame(data)


# Тестовый запуск
if __name__ == "__main__":
    client = FuelPriceClient()
    print("=== Средние цены по РФ ===")
    print(client.get_average_prices_rf())
    print("\n=== Цены в Москве ===")
    print(client.get_prices_by_region("Москва"))