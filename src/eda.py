"""
Exploratory Data Analysis (EDA) Module
Generates: statistical summary, distributions, correlation heatmap,
histograms, box plots, scatter plots, and prints business insights.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style("whitegrid")
PINK = "#F8B8D0"
DARK_PINK = "#E75480"
PALETTE = ["#F8B8D0", "#F6A6C1", "#F48FB1", "#E75480", "#C2185B"]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_PATH = os.path.join(PROJECT_ROOT, "data", "electricity_dataset_clean.csv")
OUT_DIR = os.path.join(PROJECT_ROOT, "reports", "eda")
os.makedirs(OUT_DIR, exist_ok=True)

TARGET = "Daily_Electricity_Consumption_kWh"


def run_eda():
    df = pd.read_csv(CLEAN_PATH)
    numeric_df = df.select_dtypes(include="number")

    # 1. Statistical summary
    summary = numeric_df.describe().T
    summary.to_csv(f"{OUT_DIR}/statistical_summary.csv")

    # 2. Correlation heatmap
    plt.figure(figsize=(14, 10))
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=False, cmap="RdPu", center=0, linewidths=0.4)
    plt.title("Correlation Heatmap - Electricity Consumption Features", color=DARK_PINK, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/correlation_heatmap.png", dpi=120)
    plt.close()

    # 3. Target distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df[TARGET], kde=True, color=DARK_PINK)
    plt.title("Distribution of Daily Electricity Consumption (kWh)", color=DARK_PINK, fontweight="bold")
    plt.xlabel("kWh")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/target_distribution.png", dpi=120)
    plt.close()

    # 4. Histograms of key numeric features
    key_features = ["AC_Usage_Hours", "Fan_Usage_Hours", "Lighting_Hours",
                     "Family_Members", "Outdoor_Temperature_C"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    for i, col in enumerate(key_features):
        sns.histplot(df[col], kde=True, ax=axes[i], color=PALETTE[i % len(PALETTE)])
        axes[i].set_title(col)
    fig.delaxes(axes[-1])
    plt.suptitle("Feature Distributions", color=DARK_PINK, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/feature_histograms.png", dpi=120)
    plt.close()

    # 5. Box plots (outlier visualization) by Season
    plt.figure(figsize=(9, 6))
    sns.boxplot(data=df, x="Season", y=TARGET, palette=PALETTE)
    plt.title("Electricity Consumption by Season", color=DARK_PINK, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/boxplot_season.png", dpi=120)
    plt.close()

    # 6. Box plot by House Type
    plt.figure(figsize=(9, 6))
    sns.boxplot(data=df, x="House_Type", y=TARGET, palette=PALETTE)
    plt.title("Electricity Consumption by House Type", color=DARK_PINK, fontweight="bold")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/boxplot_housetype.png", dpi=120)
    plt.close()

    # 7. Scatter plots: key relationships
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    sns.scatterplot(data=df, x="AC_Usage_Hours", y=TARGET, hue="Season", ax=axes[0], palette=PALETTE, alpha=0.6)
    axes[0].set_title("AC Hours vs Consumption")
    sns.scatterplot(data=df, x="Outdoor_Temperature_C", y=TARGET, ax=axes[1], color=DARK_PINK, alpha=0.6)
    axes[1].set_title("Outdoor Temp vs Consumption")
    sns.scatterplot(data=df, x="Family_Members", y=TARGET, ax=axes[2], color=DARK_PINK, alpha=0.6)
    axes[2].set_title("Family Size vs Consumption")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/scatter_relationships.png", dpi=120)
    plt.close()

    # 8. Business insights (auto-generated from correlations)
    target_corr = corr[TARGET].drop(TARGET).sort_values(ascending=False)
    top_positive = target_corr.head(3)
    top_negative = target_corr.tail(3)

    insights = []
    insights.append("EXPLORATORY DATA ANALYSIS - BUSINESS INSIGHTS")
    insights.append("=" * 55)
    insights.append(f"\nDataset size after cleaning: {df.shape[0]} records, {df.shape[1]} columns")
    insights.append(f"\nAverage daily consumption: {df[TARGET].mean():.2f} kWh")
    insights.append(f"Median daily consumption: {df[TARGET].median():.2f} kWh")
    insights.append(f"Std deviation: {df[TARGET].std():.2f} kWh")

    insights.append("\nTop factors INCREASING electricity consumption:")
    for feat, val in top_positive.items():
        insights.append(f"  - {feat}: correlation = {val:.3f}")

    insights.append("\nFactors most associated with LOWER consumption:")
    for feat, val in top_negative.items():
        insights.append(f"  - {feat}: correlation = {val:.3f}")

    season_avg = df.groupby("Season")[TARGET].mean().sort_values(ascending=False)
    insights.append(f"\nHighest consuming season: {season_avg.index[0]} (avg {season_avg.iloc[0]:.2f} kWh/day)")
    insights.append(f"Lowest consuming season: {season_avg.index[-1]} (avg {season_avg.iloc[-1]:.2f} kWh/day)")

    house_avg = df.groupby("House_Type")[TARGET].mean().sort_values(ascending=False)
    insights.append(f"\nHighest consuming house type: {house_avg.index[0]} (avg {house_avg.iloc[0]:.2f} kWh/day)")

    insights.append("\nBusiness Implication: AC usage and outdoor temperature are the")
    insights.append("dominant drivers of consumption, suggesting energy-saving campaigns")
    insights.append("should focus on cooling appliance efficiency during summer/monsoon months.")

    with open(f"{OUT_DIR}/business_insights.txt", "w") as f:
        f.write("\n".join(insights))

    print("\n".join(insights))
    print(f"\nEDA artifacts saved to: {OUT_DIR}")


if __name__ == "__main__":
    run_eda()
