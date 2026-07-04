# Dataset Documentation

## Overview
This dataset was **originally created** for the TEYZIX CORE Task ML-3 assignment.
No public dataset (Kaggle, UCI, GitHub, or any other source) was used, in
compliance with task requirements.

## Collection Method
**Realistic Synthetic Data Generation** — the dataset was generated
programmatically using domain-driven simulation rules based on real-world
Pakistani household electricity usage patterns (appliance wattage estimates,
seasonal temperature effects, occupancy behavior, and weekday/weekend patterns),
combined with randomized noise to mimic natural variability seen in real
smart-meter data.

Script: `data/generate_dataset.py`

## Size
- **1,212** raw records generated → **1,200** after cleaning (duplicates removed)
- **19 input features + 1 target variable**

## Feature Description

| Feature | Type | Description |
|---|---|---|
| House_Type | Categorical | Apartment / Independent House / Bungalow / Portion |
| Family_Members | Numeric | Number of people living in the house (2-10) |
| Number_of_Rooms | Numeric | Total rooms in the house |
| AC_Units | Numeric | Number of air conditioner units installed |
| AC_Usage_Hours | Numeric | Daily AC usage in hours |
| Fan_Usage_Hours | Numeric | Daily fan usage in hours |
| Refrigerator_Usage_Hours | Numeric | Daily fridge usage (constant ~24h) |
| Washing_Machine_Usage_Hours | Numeric | Daily washing machine usage |
| Water_Motor_Usage_Hours | Numeric | Daily water motor usage |
| Lighting_Hours | Numeric | Daily lighting usage |
| TV_Usage_Hours | Numeric | Daily television usage |
| Iron_Usage_Hours | Numeric | Daily iron usage |
| Kitchen_Appliance_Hours | Numeric | Combined oven/microwave/kettle usage |
| Daily_Appliance_Count | Numeric | Number of distinct appliances used that day |
| Outdoor_Temperature_C | Numeric | Outdoor temperature in Celsius |
| Day_of_Week | Categorical | Monday–Sunday |
| Is_Weekend | Binary | 1 if Saturday/Sunday |
| Season | Categorical | Summer / Winter / Spring / Monsoon |
| Is_Holiday | Binary | 1 if public holiday |
| **Daily_Electricity_Consumption_kWh** | **Target (Numeric)** | **Total daily electricity consumption** |

## Data Quality Notes
- ~2% missing values were intentionally injected into 3 columns to simulate
  real-world data collection gaps (handled via median imputation).
- ~1% duplicate rows injected and removed during preprocessing.
- A small number of extreme outliers injected and handled via IQR capping.

## Realism Assumptions
- AC/fan usage scales with outdoor temperature and season.
- Bungalows have the highest consumption multiplier; apartments/rented
  portions have lower multipliers.
- Weekend/holiday flags slightly increase appliance usage (more time at home).
- Target variable computed from an approximate wattage-based load model,
  not purely random values — ensuring genuine feature-target correlations
  exist for the ML models to learn from.
