import requests
import pandas as pd
from bs4 import BeautifulSoup
from typing import Optional
import logging
import re
import time

logger = logging.getLogger(__name__)


class FuelPriceClient:
    
    def __init__(self):
        self.session = requests.Session()
        # Более реалистичные заголовки
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        })

    def get_average_prices_rf(self) -> pd.DataFrame:
        """Получает средние цены на топливо по РФ. Всегда возвращает DataFrame."""
        
        # Пробуем спарсить
        prices = self._try_parse_benzin_price()
        
        # Если не вышло — пробуем альтернативный источник
        if not prices:
            logger.info("⚡ Пробуем альтернативный источник...")
            prices = self._try_parse_gasolina()
        
        # Если всё ещё пусто — возвращаем статические данные
        if not prices:
            logger.warning("⚠ Используем резервные данные")
            return self._get_fallback_prices()
        
        return pd.DataFrame(list(prices.items()), columns=["fuel_type", "price_rub_l"])

    def _try_parse_benzin_price(self) -> Optional[dict]:
        """Парсинг benzin-price.ru с обработкой блокировок"""
        url = "https://benzin-price.ru/"
        
        for attempt in range(3):
            try:
                response = self.session.get(url, timeout=15)
                
                # Проверяем, не вернули ли нам капчу или блок
                if response.status_code != 200 or len(response.content) < 5000:
                    logger.warning(f"⚠ Попытка #{attempt+1}: странный ответ ({response.status_code})")
                    time.sleep(2 ** attempt)  # Экспоненциальная задержка
                    continue
                
                # Пробуем разные кодировки
                for encoding in ["windows-1251", "utf-8", "cp1251"]:
                    try:
                        response.encoding = encoding
                        text = response.text
                        
                        # Проверяем, есть ли русские слова
                        if "Аи-92" in text or "Аи-95" in text:
                            return self._extract_prices_from_html(text)
                    except:
                        continue
                
                logger.warning(f"⚠ Попытка #{attempt+1}: не удалось декодировать")
                time.sleep(2)
                
            except requests.RequestException as e:
                logger.warning(f"⚠ Попытка #{attempt+1}: ошибка сети — {e}")
                time.sleep(2 ** attempt)
        
        return None

    def _extract_prices_from_html(self, html: str) -> Optional[dict]:
        """Извлекает цены из HTML-строки"""
        soup = BeautifulSoup(html, 'lxml')
        prices = {}
        
        # Ищем таблицы с ячейками class="pr"
        for table in soup.find_all('table'):
            rows = table.find_all('tr')
            if len(rows) < 2:
                continue
            
            # Проверяем заголовок
            header = rows[0]
            cells = header.find_all(['th', 'td'], class_='pr')
            if len(cells) < 4:
                continue
            
            headers = [c.get_text(strip=True).lower() for c in cells]
            if 'среднее' not in headers:
                continue
            
            avg_idx = headers.index('среднее')
            
            # Парсим строки
            for row in rows[1:]:
                data_cells = row.find_all(['td', 'th'], class_='pr')
                if len(data_cells) <= avg_idx:
                    continue
                
                # Тип топлива
                fuel_cell = row.find(['th', 'td'], class_='pr')
                if not fuel_cell:
                    continue
                fuel_text = fuel_cell.get_text(strip=True).lower()
                
                # Маппинг
                fuel_key = None
                if re.match(r'^92[\+\s]*$', fuel_text):
                    fuel_key = 'ai92'
                elif re.match(r'^95[\+\s]*$', fuel_text):
                    fuel_key = 'ai95'
                elif re.match(r'^98[\+\s]*$', fuel_text):
                    fuel_key = 'ai98'
                elif fuel_text in ['дт', 'дт+', 'дизель']:
                    fuel_key = 'diesel'
                elif fuel_text in ['газ', 'пропан']:
                    fuel_key = 'gas'
                
                if not fuel_key or fuel_key in prices:
                    continue
                
                # Цена
                avg_cell = data_cells[avg_idx]
                match = re.search(r'(\d+[\.,]\d+)', avg_cell.get_text(strip=True))
                if match:
                    prices[fuel_key] = float(match.group(1).replace(',', '.'))
                    logger.info(f"✓ {fuel_key}: {prices[fuel_key]} руб/л")
        
        return prices if prices else None

    def _try_parse_gasolina(self) -> Optional[dict]:
        """Альтернативный источник: gasolina.ru"""
        url = "https://gasolina.ru/"
        try:
            response = self.session.get(url, timeout=10)
            response.encoding = "utf-8"
            soup = BeautifulSoup(response.text, 'lxml')
            
            prices = {}
            # Простой поиск по тексту
            text = soup.get_text()
            
            patterns = {
                'ai92': r'[Аа][Ии]-?92[^\d]*(\d+[\.,]\d+)',
                'ai95': r'[Аа][Ии]-?95[^\d]*(\d+[\.,]\d+)',
                'diesel': r'[Дд][Тт]|[Дд]изель[^\d]*(\d+[\.,]\d+)',
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, text)
                if match:
                    prices[key] = float(match.group(1).replace(',', '.'))
            
            return prices if prices else None
        except:
            return None

    def _get_fallback_prices(self) -> pd.DataFrame:
        """Статические данные (обновлять вручную)"""
        logger.info("→ Используем статические цены (май 2026)")
        return pd.DataFrame([
            {"fuel_type": "ai92", "price_rub_l": 64.54},
            {"fuel_type": "ai95", "price_rub_l": 68.80},
            {"fuel_type": "ai98", "price_rub_l": 87.88},
            {"fuel_type": "diesel", "price_rub_l": 78.05},
            {"fuel_type": "gas", "price_rub_l": 31.15},
        ])

    def get_prices_by_region(self, region_name: str = "Москва") -> pd.DataFrame:
        """Цены с региональным коэффициентом"""
        df = self.get_average_prices_rf()
        
        coeffs = {
            "Москва": 1.0,
            "Санкт-Петербург": 1.02,
            "Владивосток": 1.15,
            "Новосибирск": 0.98,
            "Екатеринбург": 1.01,
        }
        coeff = coeffs.get(region_name, 1.05)
        df["price_rub_l"] = (df["price_rub_l"] * coeff).round(2)
        df["region"] = region_name
        return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    client = FuelPriceClient()
    
    print("🔍 Загрузка цен...")
    result = client.get_average_prices_rf()
    print("\n📊 Цены на топливо (руб/л):")
    print(result.to_string(index=False))