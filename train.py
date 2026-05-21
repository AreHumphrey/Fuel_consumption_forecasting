import argparse
import logging
from src.model import run_training

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)


def parse_args():
    parser = argparse.ArgumentParser(description="Обучение модели расхода топлива (РФ)")
    parser.add_argument("--data", required=True, help="Путь к подготовленному CSV")
    parser.add_argument("--output", required=True, help="Путь для сохранения модели .cbm")
    parser.add_argument("--log", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.getLogger().setLevel(args.log)
    
    logging.info(f"Начало обучения. Данные: {args.data}")
    model, metrics = run_training(args.data, args.output)
    
    logging.info("Обучение завершено!")
    logging.info(f"Метрики: {metrics}")
    logging.info(f"Модель сохранена: {args.output}")