"""
Original Synthetic Dataset Generator
Smart Electricity Consumption Prediction & Energy Optimization System

This script generates a REALISTIC, ORIGINAL synthetic dataset simulating
household electricity consumption patterns (Pakistani household context).
No public dataset (Kaggle/UCI/GitHub) is used anywhere in this project.

Method: Realistic Synthetic Data Generation using domain-driven rules
(appliance wattage assumptions, seasonal effects, occupancy behavior)
with added Gaussian noise to mimic real-world randomness.
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)

N_RECORDS = 1200  # well above the required minimum of 500

house_types = ["Apartment", "Independent House", "Bungalow", "Portion/Rented Unit"]
seasons = ["Summer", "Winter", "Spring", "Monsoon"]
days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

house_type_weights = [0.30, 0.28, 0.12, 0.30]

rows = []
for i in range(N_RECORDS):
    house_type = np.random.choice(house_types, p=house_type_weights)
    family_members = np.random.randint(2, 10)
    num_rooms = np.clip(int(np.random.normal(4, 1.5)), 1, 10)

    season = np.random.choice(seasons, p=[0.30, 0.25, 0.20, 0.25])
    day = np.random.choice(days_of_week)
    is_weekend = 1 if day in ["Saturday", "Sunday"] else 0
    is_holiday = np.random.choice([0, 1], p=[0.90, 0.10])

    # Outdoor temperature depends on season (Celsius, Pakistan climate)
    if season == "Summer":
        outdoor_temp = np.round(np.random.normal(38, 4), 1)
    elif season == "Winter":
        outdoor_temp = np.round(np.random.normal(12, 4), 1)
    elif season == "Monsoon":
        outdoor_temp = np.round(np.random.normal(31, 3), 1)
    else:  # Spring
        outdoor_temp = np.round(np.random.normal(25, 3), 1)

    # Appliance usage hours - influenced by temperature, family size, weekend
    ac_units = np.random.randint(0, min(4, num_rooms + 1))
    if season in ["Summer", "Monsoon"] and outdoor_temp > 28:
        ac_hours = np.clip(np.random.normal(7, 2.5) + (1 if is_weekend else 0), 0, 18) if ac_units > 0 else 0
    else:
        ac_hours = np.clip(np.random.normal(1, 1.2), 0, 6) if ac_units > 0 else 0

    fan_hours = np.clip(np.random.normal(10, 3) + (outdoor_temp - 25) * 0.15, 0, 24)
    fridge_hours = 24  # always on
    washing_machine_hours = np.clip(np.random.exponential(0.7) + (0.5 if is_weekend else 0), 0, 4)
    water_motor_hours = np.clip(np.random.normal(1.2, 0.6), 0, 5)
    lighting_hours = np.clip(np.random.normal(6, 2) + (2 if season == "Winter" else 0), 1, 14)
    tv_hours = np.clip(np.random.normal(4, 2) + (1.5 if is_weekend or is_holiday else 0), 0, 14)
    iron_usage_hours = np.round(np.clip(np.random.exponential(0.3), 0, 2), 2)
    kitchen_appliance_hours = np.clip(np.random.normal(2, 1), 0, 6)  # oven/microwave/kettle combined

    daily_appliance_usage_count = np.random.randint(5, 15)  # number of distinct appliances used that day

    # Approx wattage-based load simulation (kW)
    ac_load = ac_hours * ac_units * 1.5
    fan_load = fan_hours * 0.075
    fridge_load = fridge_hours * 0.15
    wm_load = washing_machine_hours * 0.5
    motor_load = water_motor_hours * 0.75
    light_load = lighting_hours * num_rooms * 0.02
    tv_load = tv_hours * 0.12
    iron_load = iron_usage_hours * 1.0
    kitchen_load = kitchen_appliance_hours * 1.2
    base_load = family_members * 0.35  # misc devices: phones, chargers, laptops

    total_kwh = (ac_load + fan_load + fridge_load + wm_load + motor_load +
                 light_load + tv_load + iron_load + kitchen_load + base_load)

    # noise for real-world randomness + house type multiplier
    house_multiplier = {"Apartment": 0.85, "Independent House": 1.0,
                         "Bungalow": 1.35, "Portion/Rented Unit": 0.75}[house_type]
    total_kwh = total_kwh * house_multiplier + np.random.normal(0, 1.5)
    total_kwh = round(max(total_kwh, 2.0), 2)

    rows.append({
        "House_Type": house_type,
        "Family_Members": family_members,
        "Number_of_Rooms": num_rooms,
        "AC_Units": ac_units,
        "AC_Usage_Hours": round(ac_hours, 2),
        "Fan_Usage_Hours": round(fan_hours, 2),
        "Refrigerator_Usage_Hours": fridge_hours,
        "Washing_Machine_Usage_Hours": round(washing_machine_hours, 2),
        "Water_Motor_Usage_Hours": round(water_motor_hours, 2),
        "Lighting_Hours": round(lighting_hours, 2),
        "TV_Usage_Hours": round(tv_hours, 2),
        "Iron_Usage_Hours": iron_usage_hours,
        "Kitchen_Appliance_Hours": round(kitchen_appliance_hours, 2),
        "Daily_Appliance_Count": daily_appliance_usage_count,
        "Outdoor_Temperature_C": outdoor_temp,
        "Day_of_Week": day,
        "Is_Weekend": is_weekend,
        "Season": season,
        "Is_Holiday": is_holiday,
        "Daily_Electricity_Consumption_kWh": total_kwh
    })

df = pd.DataFrame(rows)

# --- inject a small % missing values + duplicates + outliers (realistic + gives preprocessing something to do) ---
rng = np.random.default_rng(7)

# missing values (~2% in a few columns)
for col in ["AC_Usage_Hours", "Water_Motor_Usage_Hours", "Outdoor_Temperature_C"]:
    miss_idx = rng.choice(df.index, size=int(0.02 * len(df)), replace=False)
    df.loc[miss_idx, col] = np.nan

# duplicate rows (~1%)
dup_rows = df.sample(n=int(0.01 * len(df)), random_state=1)
df = pd.concat([df, dup_rows], ignore_index=True)

# a few extreme outliers in target
outlier_idx = rng.choice(df.index, size=6, replace=False)
df.loc[outlier_idx, "Daily_Electricity_Consumption_kWh"] *= rng.uniform(2.5, 3.5, size=6)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "electricity_dataset.csv")
df.to_csv(OUT_PATH, index=False)
print(f"Dataset generated: {df.shape[0]} rows, {df.shape[1]} columns")
print(df.head())
