import pandas as pd
import numpy as np
from pathlib import Path


def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def save_data(df: pd.DataFrame, filepath: str):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False, encoding="utf-8-sig")


def normalize_column(df: pd.DataFrame, col: str, method: str = "minmax") -> pd.DataFrame:
    df = df.copy()
    if method == "minmax":
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-8)
    elif method == "zscore":
        df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
    return df


def encode_categorical(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        df[col] = df[col].astype("category").cat.codes
    return df