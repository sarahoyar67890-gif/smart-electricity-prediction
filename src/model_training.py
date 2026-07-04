"""
Feature Engineering + Model Training + Model Evaluation Module
Trains 4 regression algorithms, compares them, and saves the best one.
Includes: feature selection, importance ranking, hyperparameter tuning (bonus).
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import json
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_PATH = os.path.join(PROJECT_ROOT, "data", "electricity_dataset_clean.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

TARGET = "Daily_Electricity_Consumption_kWh"
DARK_PINK = "#E75480"

# Final feature set used for modeling (numeric + encoded categoricals)
FEATURE_COLS = [
    "Family_Members", "Number_of_Rooms", "AC_Units", "AC_Usage_Hours",
    "Fan_Usage_Hours", "Refrigerator_Usage_Hours", "Washing_Machine_Usage_Hours",
    "Water_Motor_Usage_Hours", "Lighting_Hours", "TV_Usage_Hours",
    "Iron_Usage_Hours", "Kitchen_Appliance_Hours", "Daily_Appliance_Count",
    "Outdoor_Temperature_C", "Is_Weekend", "Is_Holiday",
    "House_Type_enc", "Day_of_Week_enc", "Season_enc"
]


def load_features():
    df = pd.read_csv(CLEAN_PATH)
    X = df[FEATURE_COLS].copy()
    y = df[TARGET].copy()
    return X, y, df


def feature_importance_analysis(X, y):
    """Quick Random Forest based feature importance for feature selection insight."""
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X, y)
    importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

    plt.figure(figsize=(9, 7))
    importance.plot(kind="barh", color=DARK_PINK)
    plt.gca().invert_yaxis()
    plt.title("Feature Importance (Random Forest)", color=DARK_PINK, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/eda/feature_importance.png", dpi=120)
    plt.close()

    importance.to_csv(f"{REPORTS_DIR}/feature_importance.csv", header=["importance"])
    return importance


def train_models(X_train, X_test, y_train, y_test, scaler_needed_cols=None):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}
    trained_models = {}

    # 1. Linear Regression (needs scaled data)
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    results["Linear Regression"] = (lr, X_test_scaled, "scaled")
    trained_models["Linear Regression"] = lr

    # 2. Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    results["Random Forest"] = (rf, X_test, "raw")
    trained_models["Random Forest"] = rf

    # 3. Gradient Boosting Regressor
    gb = GradientBoostingRegressor(n_estimators=250, learning_rate=0.05, max_depth=4, random_state=42)
    gb.fit(X_train, y_train)
    results["Gradient Boosting"] = (gb, X_test, "raw")
    trained_models["Gradient Boosting"] = gb

    # 4. XGBoost Regressor (with light hyperparameter tuning - BONUS)
    xgb_base = XGBRegressor(random_state=42, objective="reg:squarederror")
    param_grid = {
        "n_estimators": [150, 300],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
    }
    grid = GridSearchCV(xgb_base, param_grid, cv=3, scoring="r2", n_jobs=-1)
    grid.fit(X_train, y_train)
    xgb_best = grid.best_estimator_
    results["XGBoost (tuned)"] = (xgb_best, X_test, "raw")
    trained_models["XGBoost (tuned)"] = xgb_best
    print(f"Best XGBoost params: {grid.best_params_}")

    return results, trained_models, scaler


def evaluate_models(results, y_test):
    metrics = {}
    predictions = {}
    for name, (model, X_test_input, mode) in results.items():
        y_pred = model.predict(X_test_input)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        metrics[name] = {"MAE": round(mae, 3), "MSE": round(mse, 3),
                          "RMSE": round(rmse, 3), "R2_Score": round(r2, 4)}
        predictions[name] = y_pred
        print(f"{name:20s} | MAE: {mae:.3f} | RMSE: {rmse:.3f} | R2: {r2:.4f}")
    return metrics, predictions


def plot_actual_vs_predicted(y_test, predictions, best_model_name):
    plt.figure(figsize=(8, 8))
    y_pred = predictions[best_model_name]
    plt.scatter(y_test, y_pred, alpha=0.5, color=DARK_PINK, edgecolor="white")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.plot(lims, lims, "--", color="#888888", label="Perfect Prediction")
    plt.xlabel("Actual Consumption (kWh)")
    plt.ylabel("Predicted Consumption (kWh)")
    plt.title(f"Actual vs Predicted - {best_model_name}", color=DARK_PINK, fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/eda/actual_vs_predicted.png", dpi=120)
    plt.close()

    # Model comparison bar chart
    return


def plot_model_comparison(metrics):
    names = list(metrics.keys())
    r2_vals = [metrics[n]["R2_Score"] for n in names]
    rmse_vals = [metrics[n]["RMSE"] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(names, r2_vals, color="#E75480")
    axes[0].set_title("R² Score Comparison", fontweight="bold")
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(names, rmse_vals, color="#F48FB1")
    axes[1].set_title("RMSE Comparison (lower is better)", fontweight="bold")
    axes[1].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/eda/model_comparison.png", dpi=120)
    plt.close()


def run_training_pipeline():
    X, y, df = load_features()

    print("Running feature importance analysis...")
    importance = feature_importance_analysis(X, y)
    print(importance)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\nTraining 4 models...")
    results, trained_models, scaler = train_models(X_train, X_test, y_train, y_test)

    print("\nEvaluating models...")
    metrics, predictions = evaluate_models(results, y_test)

    # Select best model by R2 score
    best_model_name = max(metrics, key=lambda k: metrics[k]["R2_Score"])
    print(f"\nBest Model: {best_model_name} (R2 = {metrics[best_model_name]['R2_Score']})")

    plot_actual_vs_predicted(y_test, predictions, best_model_name)
    plot_model_comparison(metrics)

    # Save best model + scaler + feature list + metrics
    best_model = trained_models[best_model_name]
    uses_scaled = (best_model_name == "Linear Regression")

    joblib.dump({
        "model": best_model,
        "scaler": scaler,
        "uses_scaled_input": uses_scaled,
        "feature_cols": FEATURE_COLS,
        "model_name": best_model_name
    }, f"{MODELS_DIR}/best_model.pkl")

    # Save ALL trained models too (for comparison / bonus explainability)
    joblib.dump(trained_models, f"{MODELS_DIR}/all_models.pkl")

    with open(f"{REPORTS_DIR}/model_performance_report.json", "w") as f:
        json.dump({"metrics": metrics, "best_model": best_model_name}, f, indent=2)

    metrics_df = pd.DataFrame(metrics).T
    metrics_df.to_csv(f"{REPORTS_DIR}/model_performance_report.csv")

    print(f"\nSaved best model ({best_model_name}) to {MODELS_DIR}/best_model.pkl")
    return best_model_name, metrics


if __name__ == "__main__":
    run_training_pipeline()
