import requests

url = "https://benzin-price.ru/"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "ru-RU,ru;q=0.9",
}

print("🔍 Загрузка страницы...")
r = requests.get(url, headers=headers, timeout=10)

print(f"Статус: {r.status_code}")
print(f"Длина: {len(r.content)} байт")
print(f"Заголовки ответа: {dict(r.headers).get('Content-Type')}")

# Пробуем декодировать
for enc in ["windows-1251", "utf-8", "cp1251", "iso-8859-1"]:
    try:
        r.encoding = enc
        text = r.text[:500]
        if "Аи" in text or "бензин" in text.lower():
            print(f"✅ Кодировка: {enc}")
            print(f"Фрагмент: {text[200:400]}")
            break
    except:
        print(f"❌ {enc} — ошибка")