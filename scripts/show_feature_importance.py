import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from src.config import FEATURES_NUMERIC, FEATURES_CATEGORICAL
from src.features import prepare_features
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    model_path = "models/catboost_ru_v1.cbm"
    data_path = "data/train/fuel_consumption_train.csv"
    output_dir = "reports"
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    model = CatBoostRegressor()
    model.load_model(model_path)
    
    df = pd.read_csv(data_path)
    df = prepare_features(df)
    
    importance = model.get_feature_importance()
    
    importance_df = pd.DataFrame({
        "feature": FEATURES_NUMERIC + FEATURES_CATEGORICAL,
        "importance": importance
    }).sort_values("importance", ascending=False)
    
    print("\n" + "="*60)
    print("ТОП-15 ВАЖНЫХ ПРИЗНАКОВ")
    print("="*60)
    for i, row in importance_df.head(15).iterrows():
        print(f"{i+1:2d}. {row['feature']:25s} : {row['importance']:6.2f}%")
    print("="*60)
    
    plt.figure(figsize=(12, 10))
    sns.barplot(data=importance_df.head(15), x='importance', y='feature', palette='viridis')
    plt.xlabel('Важность признака (%)')
    plt.ylabel('Признак')
    plt.title('Важность признаков модели')
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    importance_df.to_csv(Path(output_dir) / 'feature_importance.csv', index=False)
    print(f"\nГрафик сохранен: {output_dir}/feature_importance.png")

if __name__ == "__main__":
    main()