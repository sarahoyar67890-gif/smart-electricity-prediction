"""
Smart Electricity Consumption Prediction & Energy Optimization System
Streamlit Prediction Interface — Electric Circuit Aesthetic Edition (v2)

Bonus features implemented in this file:
  - Explainable AI (SHAP)
  - Real-time prediction
  - Electricity cost calculator (progressive slab estimator)
  - Interactive multi-tab dashboard
  - What-If optimization comparison (unique)
  - Appliance-by-appliance load breakdown gauge + chart (unique)
  - LIVE WEATHER API integration (Open-Meteo, free, no API key)

Run with: streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import os
import requests

st.set_page_config(page_title="ElectroPredict | Energy Optimizer",
                    page_icon="⚡", layout="wide",
                    initial_sidebar_state="collapsed")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "electricity_dataset_clean.csv")

# ============================================================
# THEME: Electric Circuit v2 — deep indigo/violet base, neon gold
# and electric-pink accents, animated circuit-line dividers,
# glowing pulse effects to evoke "electricity flowing".
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Poppins:wght@300;400;500;600;700&display=swap');

* { font-family: 'Poppins', sans-serif; }

.stApp {
    background-color: #120E2A;
    background-image:
        radial-gradient(circle at 15% 20%, rgba(255, 92, 158, 0.18) 0%, transparent 35%),
        radial-gradient(circle at 85% 15%, rgba(255, 200, 87, 0.14) 0%, transparent 35%),
        radial-gradient(circle at 50% 90%, rgba(122, 92, 255, 0.20) 0%, transparent 45%),
        linear-gradient(160deg, #14103A 0%, #1B1547 50%, #23124F 100%),
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='120' viewBox='0 0 120 120'%3E%3Cg fill='none' stroke='%23FF5C9E' stroke-opacity='0.07' stroke-width='1.4'%3E%3Cpath d='M10 10h30v30h30v-20h40'/%3E%3Ccircle cx='10' cy='10' r='3'/%3E%3Ccircle cx='40' cy='40' r='3'/%3E%3Ccircle cx='70' cy='20' r='3'/%3E%3Cpath d='M10 90h25v-25h50v40h30'/%3E%3Ccircle cx='10' cy='90' r='3'/%3E%3Ccircle cx='60' cy='65' r='3'/%3E%3Ccircle cx='110' cy='105' r='3'/%3E%3C/g%3E%3C/svg%3E");
    background-size: cover, cover, cover, cover, 240px 240px;
    background-attachment: fixed;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {background: transparent !important;}

.hero-banner {
    background: linear-gradient(120deg, #2A1B6B 0%, #6B2E8C 45%, #C22E7A 100%);
    border-radius: 26px;
    padding: 40px 44px;
    margin-bottom: 26px;
    box-shadow: 0 0 40px rgba(255, 92, 158, 0.35), 0 0 80px rgba(122, 92, 255, 0.25);
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(255, 200, 87, 0.25);
}
.hero-banner::before {
    content: "⚡"; position: absolute; top: -30px; right: 10px;
    font-size: 180px; opacity: 0.08; transform: rotate(15deg);
}
.hero-title {
    color: #FFFFFF; font-family: 'Orbitron', sans-serif;
    font-weight: 900; font-size: 38px; margin: 0; letter-spacing: 1px;
    text-shadow: 0 0 18px rgba(255, 200, 87, 0.6), 0 0 6px rgba(255,255,255,0.4);
    position: relative; z-index: 1;
    animation: pulseGlow 3s ease-in-out infinite;
}
@keyframes pulseGlow {
    0%, 100% { text-shadow: 0 0 18px rgba(255, 200, 87, 0.6), 0 0 6px rgba(255,255,255,0.4); }
    50% { text-shadow: 0 0 30px rgba(255, 200, 87, 0.95), 0 0 12px rgba(255,255,255,0.7); }
}
.hero-subtitle {
    color: #E8DBFF; font-size: 16px; margin-top: 10px;
    font-weight: 400; position: relative; z-index: 1;
}
.hero-badge {
    display: inline-block; background: rgba(255, 200, 87, 0.18);
    color: #FFD866; padding: 6px 18px; border-radius: 30px;
    font-size: 13px; font-weight: 600; margin-top: 16px; margin-right: 8px;
    border: 1px solid rgba(255, 200, 87, 0.4);
    position: relative; z-index: 1;
}
.appliance-strip {
    margin-top: 18px; font-size: 22px; letter-spacing: 14px;
    opacity: 0.85; position: relative; z-index: 1;
}

/* ---------- Animated circuit-line divider ---------- */
.circuit-divider {
    height: 3px; margin: 10px 0 22px 0; border-radius: 3px;
    background: linear-gradient(90deg, transparent, #FF5C9E, #FFC857, #7A5CFF, transparent);
    background-size: 200% 100%;
    animation: flowLine 4s linear infinite;
}
@keyframes flowLine {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}

h1, h2, h3 { color: #FFD866 !important; font-family: 'Orbitron', sans-serif; font-weight: 700 !important; }
h3 { border-left: 4px solid #FF5C9E; padding-left: 12px; }
p, span, label, .stMarkdown { color: #E8DBFF; }

.stTabs [data-baseweb="tab-list"] {
    gap: 8px; background: rgba(255,255,255,0.05); padding: 8px;
    border-radius: 18px; border: 1px solid rgba(255,200,87,0.15);
}
.stTabs [data-baseweb="tab"] {
    height: 48px; border-radius: 14px; font-weight: 600; color: #C9B8FF;
    background-color: transparent; padding: 0 20px;
    transition: all 0.25s ease;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #FF5C9E, #7A5CFF) !important;
    color: white !important;
    box-shadow: 0 0 18px rgba(255, 92, 158, 0.5);
}

.stButton>button {
    background: linear-gradient(90deg, #FFC857, #FF5C9E);
    color: #1B1130; border-radius: 24px; border: none;
    padding: 0.7em 2em; font-weight: 700; font-size: 16px;
    box-shadow: 0 0 20px rgba(255, 92, 158, 0.45);
    transition: all 0.25s ease;
}
.stButton>button:hover {
    transform: translateY(-2px) scale(1.02);
    box-shadow: 0 0 30px rgba(255, 200, 87, 0.7);
}
.stButton>button:active { transform: translateY(0) scale(0.98); }

div[data-testid="stMetric"] {
    background: linear-gradient(160deg, rgba(255,255,255,0.07) 0%, rgba(122,92,255,0.10) 100%);
    border-radius: 18px; padding: 18px 14px;
    border: 1px solid rgba(255, 200, 87, 0.25);
    box-shadow: 0 0 16px rgba(122, 92, 255, 0.18);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 0 24px rgba(255, 92, 158, 0.35);
}
div[data-testid="stMetricLabel"] { color: #C9B8FF !important; font-weight: 600 !important; }
div[data-testid="stMetricValue"] { color: #FFD866 !important; font-weight: 800 !important; }

.pink-card {
    background: linear-gradient(160deg, rgba(255,255,255,0.06) 0%, rgba(255,92,158,0.08) 100%);
    border-radius: 16px; padding: 18px 22px;
    border: 1px solid rgba(255, 92, 158, 0.25);
    box-shadow: 0 0 14px rgba(255, 92, 158, 0.12);
    margin-bottom: 14px; font-size: 15px; color: #F1E8FF;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.pink-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 22px rgba(255, 92, 158, 0.28);
    border-color: rgba(255, 200, 87, 0.4);
}
.section-card {
    background: rgba(255,255,255,0.04);
    border-radius: 22px; padding: 24px 28px; margin-bottom: 22px;
    border: 1px solid rgba(122, 92, 255, 0.25);
    box-shadow: 0 0 20px rgba(122, 92, 255, 0.12);
}
.weather-card {
    background: linear-gradient(135deg, rgba(87, 200, 255, 0.14), rgba(122, 92, 255, 0.14));
    border-radius: 18px; padding: 20px 24px; margin-bottom: 16px;
    border: 1px solid rgba(87, 200, 255, 0.35);
    box-shadow: 0 0 18px rgba(87, 200, 255, 0.2);
}
.savings-banner {
    background: linear-gradient(90deg, rgba(255,200,87,0.18), rgba(255,92,158,0.18));
    border-radius: 16px; padding: 16px 22px; margin-top: 10px;
    border-left: 4px solid #FFC857; color: #FFF3D6; font-weight: 600;
    box-shadow: 0 0 18px rgba(255, 200, 87, 0.2);
}
.gauge-wrap { text-align: center; padding: 6px 0 0 0; }

div[data-testid="stSlider"] label { color: #C9B8FF; font-weight: 600; }
.stSlider [role="slider"] { background-color: #FF5C9E !important; }

div[data-baseweb="select"] > div {
    border-radius: 14px !important; border: 1px solid rgba(255,200,87,0.3) !important;
    background-color: rgba(30, 20, 60, 0.85) !important;
}
div[data-baseweb="select"] * { color: #F1E8FF !important; }
div[data-baseweb="popover"] { background-color: #1E1440 !important; }
div[data-baseweb="popover"] * { color: #F1E8FF !important; }
ul[role="listbox"] { background-color: #1E1440 !important; }
li[role="option"] { background-color: #1E1440 !important; color: #F1E8FF !important; }
li[role="option"]:hover { background-color: rgba(255, 92, 158, 0.25) !important; }

div[data-testid="stCheckbox"] label p { color: #E8DBFF !important; }
div[data-testid="stSlider"] div[data-testid="stTickBarMin"],
div[data-testid="stSlider"] div[data-testid="stTickBarMax"] { color: #C9B8FF !important; }
.stMarkdown p, .stMarkdown li { color: #E8DBFF; }
[data-testid="stCaptionContainer"] { color: #C9B8FF !important; }

div[data-testid="stTextInput"] input {
    background-color: rgba(30, 20, 60, 0.85) !important;
    color: #F1E8FF !important;
    border: 1px solid rgba(255,200,87,0.3) !important;
    border-radius: 14px !important;
}

.streamlit-expanderHeader {
    background: rgba(255,255,255,0.05); border-radius: 14px;
    font-weight: 600; color: #FFD866;
}

[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; }

.footer-pink {
    text-align: center; color: #C9B8FF; padding: 20px 0 8px 0;
    font-size: 14px; font-weight: 500;
}

hr { border-top: 1px solid rgba(255, 200, 87, 0.2) !important; }
</style>
""", unsafe_allow_html=True)

# ----------------------------- LOAD ARTIFACTS -----------------------------
@st.cache_resource
def load_model():
    return joblib.load(f"{MODELS_DIR}/best_model.pkl")

@st.cache_resource
def load_encoders():
    return joblib.load(f"{MODELS_DIR}/label_encoders.pkl")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

model_bundle = load_model()
encoders = load_encoders()
df_ref = load_data()

model = model_bundle["model"]
scaler = model_bundle["scaler"]
uses_scaled = model_bundle["uses_scaled_input"]
feature_cols = model_bundle["feature_cols"]
model_name = model_bundle["model_name"]

RATE_SLABS = [
    (100, 15), (200, 22), (300, 28), (700, 35), (float("inf"), 45)
]  # (upto units, PKR/unit) - simplified progressive slab estimate


def estimate_bill(monthly_units):
    remaining = monthly_units
    prev_cap = 0
    total = 0
    for cap, rate in RATE_SLABS:
        slab_units = min(remaining, cap - prev_cap)
        if slab_units <= 0:
            break
        total += slab_units * rate
        remaining -= slab_units
        prev_cap = cap
        if remaining <= 0:
            break
    return round(total, 0)


def efficiency_score(predicted_kwh, family_members, rooms):
    per_capita = predicted_kwh / max(family_members, 1)
    baseline = 6.0
    score = 100 - min(100, max(0, (per_capita - baseline) * 8))
    return round(score, 1)


@st.cache_data(ttl=600, show_spinner=False)
def fetch_live_weather(city_name):
    """BONUS: Live Weather API integration using Open-Meteo (free, no API key required)."""
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
    geo_resp = requests.get(geo_url, params={"name": city_name, "count": 1}, timeout=8)
    geo_data = geo_resp.json()
    if not geo_data.get("results"):
        return None
    loc = geo_data["results"][0]
    lat, lon = loc["latitude"], loc["longitude"]
    display_name = f"{loc.get('name')}, {loc.get('country', '')}"

    weather_url = "https://api.open-meteo.com/v1/forecast"
    w_resp = requests.get(weather_url, params={
        "latitude": lat, "longitude": lon, "current": "temperature_2m,relative_humidity_2m"
    }, timeout=8)
    w_data = w_resp.json()
    current = w_data.get("current", {})
    return {
        "location": display_name,
        "temperature": current.get("temperature_2m"),
        "humidity": current.get("relative_humidity_2m"),
    }


def build_input_row(house_type, family_members, num_rooms, ac_units, ac_hours, fan_hours,
                     fridge_hours, wm_hours, motor_hours, lighting_hours, tv_hours,
                     iron_hours, kitchen_hours, appliance_count, outdoor_temp,
                     day_of_week, season, is_holiday):
    is_weekend = 1 if day_of_week in ["Saturday", "Sunday"] else 0
    input_dict = {
        "Family_Members": family_members, "Number_of_Rooms": num_rooms, "AC_Units": ac_units,
        "AC_Usage_Hours": ac_hours, "Fan_Usage_Hours": fan_hours,
        "Refrigerator_Usage_Hours": fridge_hours, "Washing_Machine_Usage_Hours": wm_hours,
        "Water_Motor_Usage_Hours": motor_hours, "Lighting_Hours": lighting_hours,
        "TV_Usage_Hours": tv_hours, "Iron_Usage_Hours": iron_hours,
        "Kitchen_Appliance_Hours": kitchen_hours, "Daily_Appliance_Count": appliance_count,
        "Outdoor_Temperature_C": outdoor_temp, "Is_Weekend": is_weekend,
        "Is_Holiday": int(is_holiday),
        "House_Type_enc": encoders["House_Type"].transform([house_type])[0],
        "Day_of_Week_enc": encoders["Day_of_Week"].transform([day_of_week])[0],
        "Season_enc": encoders["Season"].transform([season])[0],
    }
    return pd.DataFrame([input_dict])[feature_cols]


def predict_kwh(X_input):
    X_for_pred = scaler.transform(X_input) if uses_scaled else X_input
    return max(float(model.predict(X_for_pred)[0]), 0.5)


def draw_gauge(value, max_value=40, label="Daily kWh"):
    fig, ax = plt.subplots(figsize=(4.2, 2.4), subplot_kw={"aspect": "equal"})
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    frac = min(value / max_value, 1.0)
    colors_list = ["#4CD9A8", "#FFD866", "#FF9E5C", "#FF5C9E"]
    bounds = [0, 0.35, 0.6, 0.8, 1.0]
    for i in range(4):
        theta1 = 180 - bounds[i] * 180
        theta2 = 180 - bounds[i+1] * 180
        wedge = plt.matplotlib.patches.Wedge((0, 0), 1, theta2, theta1, width=0.32,
                                              facecolor=colors_list[i], edgecolor="none", alpha=0.9)
        ax.add_patch(wedge)

    needle_angle = np.radians(180 - frac * 180)
    ax.plot([0, 0.78*np.cos(needle_angle)], [0, 0.78*np.sin(needle_angle)],
            color="white", linewidth=3, solid_capstyle="round")
    ax.add_patch(plt.Circle((0, 0), 0.06, color="white", zorder=5))

    ax.text(0, -0.35, f"{value:.1f} kWh", ha="center", va="center",
            fontsize=16, fontweight="bold", color="white")
    ax.text(0, -0.58, label, ha="center", va="center", fontsize=10, color="#C9B8FF")

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.7, 1.1)
    ax.axis("off")
    return fig


def circuit_divider():
    st.markdown('<div class="circuit-divider"></div>', unsafe_allow_html=True)


# ----------------------------- HERO BANNER -----------------------------
st.markdown(f"""
<div class="hero-banner">
    <div class="hero-title">⚡ ELECTROPREDICT — Smart Energy Optimizer</div>
    <div class="hero-subtitle">AI-powered electricity forecasting for your home, appliance by appliance</div>
    <div class="hero-badge">🤖 Powered by {model_name}</div>
    <div class="hero-badge">🌦️ Live Weather Enabled</div>
    <div class="appliance-strip">❄️ 🌀 🧊 🧺 🚰 💡 📺 👕 🍳 🔌</div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "🔮  Predict My Usage", "📊  Dashboard & Insights",
    "🧠  Explainability (SHAP)", "⚖️  What-If Comparison"
])

# ============================================================
# TAB 1 — PREDICTION
# ============================================================
with tab1:
    st.markdown('<div class="weather-card">', unsafe_allow_html=True)
    st.markdown("#### 🌦️ Bonus: Auto-fill Outdoor Temperature from Live Weather")
    wcol1, wcol2 = st.columns([3, 1])
    with wcol1:
        city_input = st.text_input("Enter your city", value="Lahore", label_visibility="collapsed",
                                    placeholder="Enter your city (e.g. Lahore)")
    with wcol2:
        fetch_weather_btn = st.button("🔄 Fetch Live Weather", use_container_width=True)

    if fetch_weather_btn:
        with st.spinner("Fetching live weather..."):
            try:
                weather = fetch_live_weather(city_input)
            except Exception:
                weather = None
        if weather and weather["temperature"] is not None:
            st.session_state["live_temp"] = weather["temperature"]
            st.success(f"📍 {weather['location']}: {weather['temperature']}°C, "
                       f"{weather['humidity']}% humidity — outdoor temperature slider updated below!")
        else:
            st.warning("Couldn't fetch weather for that city. Check spelling or your internet connection, "
                       "and set the temperature manually below.")
    st.markdown('</div>', unsafe_allow_html=True)

    circuit_divider()

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### 🏡 Tell Us About Your Household")
    col1, col2, col3 = st.columns(3)

    with col1:
        house_type = st.selectbox("🏠 House Type", encoders["House_Type"].classes_)
        family_members = st.slider("👨‍👩‍👧‍👦 Family Members", 1, 12, 5)
        num_rooms = st.slider("🚪 Number of Rooms", 1, 10, 4)
        ac_units = st.slider("❄️ Number of AC Units", 0, 4, 1)
        ac_hours = st.slider("❄️ AC Usage Hours/Day", 0.0, 18.0, 4.0)
        fan_hours = st.slider("🌀 Fan Usage Hours/Day", 0.0, 24.0, 10.0)

    with col2:
        fridge_hours = st.slider("🧊 Refrigerator Hours/Day", 0.0, 24.0, 24.0)
        wm_hours = st.slider("🧺 Washing Machine Hours/Day", 0.0, 4.0, 0.7)
        motor_hours = st.slider("🚰 Water Motor Hours/Day", 0.0, 5.0, 1.2)
        lighting_hours = st.slider("💡 Lighting Hours/Day", 1.0, 14.0, 6.0)
        tv_hours = st.slider("📺 TV Usage Hours/Day", 0.0, 14.0, 4.0)
        iron_hours = st.slider("👕 Iron Usage Hours/Day", 0.0, 2.0, 0.3)

    with col3:
        kitchen_hours = st.slider("🍳 Kitchen Appliance Hours/Day", 0.0, 6.0, 2.0)
        appliance_count = st.slider("🔌 Distinct Appliances Used Today", 5, 15, 8)
        default_temp = float(st.session_state.get("live_temp", 30.0))
        outdoor_temp = st.slider("🌡️ Outdoor Temperature (°C)", 5.0, 48.0, default_temp)
        day_of_week = st.selectbox("📅 Day of Week", encoders["Day_of_Week"].classes_)
        season = st.selectbox("🍃 Season", encoders["Season"].classes_)
        is_holiday = st.checkbox("🎉 Is Holiday?")

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    _, mid_col, _ = st.columns([1, 1, 1])
    with mid_col:
        predict_btn = st.button("⚡ Predict Electricity Consumption ⚡", use_container_width=True)

    if predict_btn:
        X_input = build_input_row(house_type, family_members, num_rooms, ac_units, ac_hours,
                                   fan_hours, fridge_hours, wm_hours, motor_hours, lighting_hours,
                                   tv_hours, iron_hours, kitchen_hours, appliance_count,
                                   outdoor_temp, day_of_week, season, is_holiday)
        predicted_daily = predict_kwh(X_input)
        predicted_monthly = predicted_daily * 30
        estimated_bill = estimate_bill(predicted_monthly)
        eff_score = efficiency_score(predicted_daily, family_members, num_rooms)

        if ac_hours > 4:
            peak_hours = "1 PM – 6 PM (afternoon cooling peak)"
        elif lighting_hours > 7 or tv_hours > 6:
            peak_hours = "7 PM – 11 PM (evening peak)"
        else:
            peak_hours = "6 AM – 9 AM (morning routine peak)"

        circuit_divider()
        st.markdown("## 🎯 Your Prediction Results")

        gauge_col, metrics_col = st.columns([1, 2])
        with gauge_col:
            st.markdown('<div class="section-card gauge-wrap">', unsafe_allow_html=True)
            fig = draw_gauge(predicted_daily, max_value=40)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.markdown('</div>', unsafe_allow_html=True)

        with metrics_col:
            m1, m2 = st.columns(2)
            m1.metric("📆 Monthly Estimate", f"{predicted_monthly:.1f} kWh")
            m2.metric("💰 Est. Monthly Bill", f"PKR {estimated_bill:,.0f}")
            m3, m4 = st.columns(2)
            m3.metric("🌿 Efficiency Score", f"{eff_score}/100")
            m4.metric("⏰ Peak Hours", peak_hours.split(" (")[0])

        st.markdown("### 🔌 Appliance-by-Appliance Load Breakdown")
        appliance_loads = {
            "❄️ AC": ac_hours * ac_units * 1.5,
            "🌀 Fan": fan_hours * 0.075,
            "🧊 Fridge": fridge_hours * 0.15,
            "🧺 Washing Machine": wm_hours * 0.5,
            "🚰 Water Motor": motor_hours * 0.75,
            "💡 Lighting": lighting_hours * num_rooms * 0.02,
            "📺 TV": tv_hours * 0.12,
            "👕 Iron": iron_hours * 1.0,
            "🍳 Kitchen": kitchen_hours * 1.2,
        }
        load_df = pd.DataFrame(list(appliance_loads.items()), columns=["Appliance", "Estimated kWh"])
        load_df = load_df.sort_values("Estimated kWh", ascending=True)

        fig_load, ax_load = plt.subplots(figsize=(8, 4.5))
        fig_load.patch.set_alpha(0)
        ax_load.set_facecolor("none")
        ax_load.barh(load_df["Appliance"], load_df["Estimated kWh"], color="#FF5C9E")
        ax_load.tick_params(colors="#E8DBFF", labelsize=10)
        ax_load.spines[["top", "right"]].set_visible(False)
        ax_load.set_xlabel("Estimated kWh", color="#C9B8FF")
        for spine in ax_load.spines.values():
            spine.set_alpha(0.4)
            spine.set_color("#7A5CFF")
        st.pyplot(fig_load, use_container_width=True)
        plt.close(fig_load)

        st.markdown(f"<div class='pink-card'>⏰ <b>Estimated Peak Usage Hours:</b> {peak_hours}</div>", unsafe_allow_html=True)

        st.markdown("### 💡 Personalized Energy Optimization Tips")
        recs = []
        if ac_hours > 6:
            recs.append("❄️ Your AC usage is high — raising the thermostat by 2°C could cut consumption by 10-15%.")
        if wm_hours > 0 and not (day_of_week in ["Saturday", "Sunday"]):
            recs.append("🧺 Consider running the washing machine during off-peak hours (before 5 PM or after 10 PM).")
        if lighting_hours > 8:
            recs.append("💡 Switch to LED bulbs if not already — could reduce lighting load by up to 40%.")
        if eff_score < 50:
            recs.append("⚠️ Your efficiency score is below average — review standby/base load devices (routers, chargers).")
        if outdoor_temp > 35 and ac_units > 1:
            recs.append("🌡️ Extreme heat detected — stagger AC unit usage across rooms instead of running all simultaneously.")
        if not recs:
            recs.append("✅ Great job! Your usage pattern looks efficient. Keep monitoring peak hours to save more.")

        for r in recs:
            st.markdown(f"<div class='pink-card'>{r}</div>", unsafe_allow_html=True)

        est_savings = round(predicted_monthly * 0.15 * 45, 0)
        st.markdown(f"""
        <div class="savings-banner">
            💰 Potential estimated monthly savings if optimizations applied: <b>PKR {est_savings:,.0f}</b>
        </div>
        """, unsafe_allow_html=True)

        st.session_state["last_inputs"] = dict(
            house_type=house_type, family_members=family_members, num_rooms=num_rooms,
            ac_units=ac_units, ac_hours=ac_hours, fan_hours=fan_hours, fridge_hours=fridge_hours,
            wm_hours=wm_hours, motor_hours=motor_hours, lighting_hours=lighting_hours,
            tv_hours=tv_hours, iron_hours=iron_hours, kitchen_hours=kitchen_hours,
            appliance_count=appliance_count, outdoor_temp=outdoor_temp,
            day_of_week=day_of_week, season=season, is_holiday=is_holiday,
            predicted_daily=predicted_daily
        )

# ============================================================
# TAB 2 — DASHBOARD & INSIGHTS
# ============================================================
with tab2:
    st.markdown("### 📈 Dataset Overview")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Total Records", df_ref.shape[0])
    d2.metric("Avg Daily (kWh)", f"{df_ref['Daily_Electricity_Consumption_kWh'].mean():.2f}")
    d3.metric("Max Daily (kWh)", f"{df_ref['Daily_Electricity_Consumption_kWh'].max():.2f}")
    d4.metric("Min Daily (kWh)", f"{df_ref['Daily_Electricity_Consumption_kWh'].min():.2f}")

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("**🍃 Average Consumption by Season**")
        season_avg = df_ref.groupby("Season")["Daily_Electricity_Consumption_kWh"].mean().sort_values(ascending=False)
        st.bar_chart(season_avg, color="#FF5C9E")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("**🏠 Average Consumption by House Type**")
        house_avg = df_ref.groupby("House_Type")["Daily_Electricity_Consumption_kWh"].mean().sort_values(ascending=False)
        st.bar_chart(house_avg, color="#FFC857")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("**❄️ Consumption vs AC Usage Hours**")
    st.scatter_chart(df_ref, x="AC_Usage_Hours", y="Daily_Electricity_Consumption_kWh", color="Season")
    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("🔍 View Raw Dataset Sample"):
        st.dataframe(df_ref.head(50), use_container_width=True)

# ============================================================
# TAB 3 — SHAP EXPLAINABILITY (BONUS FEATURE)
# ============================================================
with tab3:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### 🧠 Explainable AI — Why the Model Predicts What It Predicts")
    st.caption("Bonus Feature: SHAP (SHapley Additive exPlanations) shows how each feature pushes the prediction up or down.")

    @st.cache_resource
    def get_shap_explainer():
        X_sample = df_ref[feature_cols].sample(min(150, len(df_ref)), random_state=42)
        if uses_scaled:
            X_sample_input = scaler.transform(X_sample)
            explainer = shap.Explainer(model.predict, X_sample_input)
            shap_values = explainer(X_sample_input)
        else:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X_sample)
        return shap_values, X_sample

    if st.button("🔍 Generate SHAP Explanation"):
        with st.spinner("Computing SHAP values..."):
            shap_values, X_sample = get_shap_explainer()

            fig, ax = plt.subplots(figsize=(9, 6))
            fig.patch.set_alpha(0)
            shap.summary_plot(shap_values, X_sample, show=False, plot_size=None)
            st.pyplot(fig)
            plt.close(fig)

            st.markdown("""
            <div class='pink-card'>
            📌 <b>How to read this:</b> Features at the top have the biggest impact on predictions.
            Pink dots pushing right = increases predicted consumption. Blue dots pushing left = decreases it.
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# TAB 4 — WHAT-IF COMPARISON (UNIQUE FEATURE)
# ============================================================
with tab4:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### ⚖️ What-If Optimization Comparison")
    st.caption("See how much you'd save if you cut AC and lighting usage — compared side-by-side with your last prediction.")

    if "last_inputs" not in st.session_state:
        st.info("👉 Run a prediction in the **Predict My Usage** tab first to unlock this comparison.")
    else:
        li = st.session_state["last_inputs"]

        st.markdown("Adjust the sliders below to simulate an optimized version of your household:")
        oc1, oc2 = st.columns(2)
        with oc1:
            new_ac_hours = st.slider("❄️ Optimized AC Hours/Day", 0.0, 18.0, max(li["ac_hours"] - 2, 0.0))
        with oc2:
            new_lighting_hours = st.slider("💡 Optimized Lighting Hours/Day", 1.0, 14.0, max(li["lighting_hours"] - 2, 1.0))

        X_original = build_input_row(
            li["house_type"], li["family_members"], li["num_rooms"], li["ac_units"], li["ac_hours"],
            li["fan_hours"], li["fridge_hours"], li["wm_hours"], li["motor_hours"], li["lighting_hours"],
            li["tv_hours"], li["iron_hours"], li["kitchen_hours"], li["appliance_count"],
            li["outdoor_temp"], li["day_of_week"], li["season"], li["is_holiday"])

        X_optimized = build_input_row(
            li["house_type"], li["family_members"], li["num_rooms"], li["ac_units"], new_ac_hours,
            li["fan_hours"], li["fridge_hours"], li["wm_hours"], li["motor_hours"], new_lighting_hours,
            li["tv_hours"], li["iron_hours"], li["kitchen_hours"], li["appliance_count"],
            li["outdoor_temp"], li["day_of_week"], li["season"], li["is_holiday"])

        original_daily = predict_kwh(X_original)
        optimized_daily = predict_kwh(X_optimized)
        saved_daily = max(original_daily - optimized_daily, 0)
        saved_monthly_bill = estimate_bill(saved_daily * 30)

        comp_col1, comp_col2, comp_col3 = st.columns(3)
        comp_col1.metric("🔴 Current Daily Usage", f"{original_daily:.2f} kWh")
        comp_col2.metric("🟢 Optimized Daily Usage", f"{optimized_daily:.2f} kWh")
        comp_col3.metric("💸 Monthly Savings", f"PKR {saved_monthly_bill:,.0f}")

        compare_df = pd.DataFrame({
            "Scenario": ["Current", "Optimized"],
            "Daily kWh": [original_daily, optimized_daily]
        }).set_index("Scenario")
        st.bar_chart(compare_df, color="#FFC857")

        if saved_daily > 0:
            st.markdown(f"""
            <div class="savings-banner">
                🎉 By reducing AC to {new_ac_hours:.1f} hrs and lighting to {new_lighting_hours:.1f} hrs/day,
                you could save <b>{saved_daily:.2f} kWh/day</b> — about <b>PKR {saved_monthly_bill:,.0f}/month</b>!
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Adjust the sliders to values lower than your original usage to see potential savings.")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p class='footer-pink'>⚡ Made with electric energy for TEYZIX CORE Internship — Task ML-3 ⚡</p>", unsafe_allow_html=True)
