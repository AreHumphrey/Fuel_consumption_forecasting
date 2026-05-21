import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')


def load_predictions(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def calculate_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mdae = median_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true.replace(0, np.nan))) * 100
    
    errors = y_true - y_pred
    return {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "r2": round(r2, 4),
        "mdae": round(mdae, 3),
        "mape_percent": round(mape, 2),
        "mean_error": round(errors.mean(), 3),
        "std_error": round(errors.std(), 3),
        "min_error": round(errors.min(), 3),
        "max_error": round(errors.max(), 3),
    }


def plot_predictions(y_true: pd.Series, y_pred: pd.Series, output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    errors = y_true - y_pred
    
    axes[0, 0].scatter(y_true, y_pred, alpha=0.6, edgecolors='black')
    axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Фактический расход (л/100км)')
    axes[0, 0].set_ylabel('Предсказанный расход (л/100км)')
    axes[0, 0].set_title('Факт vs Предсказание')
    axes[0, 0].grid(alpha=0.3)
    
    axes[0, 1].hist(errors, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(x=0, color='red', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Ошибка предсказания (л/100км)')
    axes[0, 1].set_ylabel('Частота')
    axes[0, 1].set_title('Распределение ошибок')
    axes[0, 1].grid(alpha=0.3)
    
    axes[1, 0].scatter(y_true, errors, alpha=0.6, edgecolors='black')
    axes[1, 0].axhline(y=0, color='red', linestyle='--', lw=2)
    axes[1, 0].set_xlabel('Фактический расход (л/100км)')
    axes[1, 0].set_ylabel('Ошибка (л/100км)')
    axes[1, 0].set_title('Ошибки по диапазону расхода')
    axes[1, 0].grid(alpha=0.3)
    
    axes[1, 1].boxplot(errors, vert=True, patch_artist=True)
    axes[1, 1].set_ylabel('Ошибка (л/100км)')
    axes[1, 1].set_title('Боксплот ошибок')
    axes[1, 1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'predictions_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def analyze_by_category(df: pd.DataFrame, target: str, pred: str, category_cols: list) -> dict:
    results = {}
    for col in category_cols:
        if col in df.columns:
            grouped_data = {}
            for name, group in df.groupby(col):
                mae_val = mean_absolute_error(group[target], group[pred])
                grouped_data[name] = {
                    "count": len(group),
                    "actual_mean": round(group[target].mean(), 3),
                    "predicted_mean": round(group[pred].mean(), 3),
                    "mae": round(mae_val, 3)
                }
            results[col] = grouped_data
    return results


def main():
    predictions_file = "data/predictions.csv"
    output_dir = "reports"
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    df = load_predictions(predictions_file)
    
    target_col = "fuel_consumption_l100km"
    pred_col = "predicted_consumption"

    metrics = calculate_metrics(df[target_col], df[pred_col])
    
    print("\n" + "="*60)

    print("="*60)
    for key, value in metrics.items():
        print(f"{key:20s}: {value}")
    print("="*60)

    print(f"  Фактический расход:  {df[target_col].mean():.2f} ± {df[target_col].std():.2f} л/100км")
    print(f"  Предсказанный расход: {df[pred_col].mean():.2f} ± {df[pred_col].std():.2f} л/100км")
    print(f"  Диапазон фактический: [{df[target_col].min():.2f}, {df[target_col].max():.2f}]")
    print(f"  Диапазон предсказанный: [{df[pred_col].min():.2f}, {df[pred_col].max():.2f}]")

    category_cols = ["vehicle_type", "fuel_type", "region_code", "season"]
    category_analysis = analyze_by_category(df, target_col, pred_col, category_cols)
    
    for col, analysis in category_analysis.items():
        print(f"\n  {col}:")
        for cat, stats in list(analysis.items())[:5]:
            print(f"    {cat}: n={stats['count']}, MAE={stats['mae']}")
    

    plot_predictions(df[target_col], df[pred_col], output_dir)
    
    metrics_path = Path(output_dir) / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    df.to_csv(Path(output_dir) / "predictions_with_errors.csv", index=False)



if __name__ == "__main__":
    main()