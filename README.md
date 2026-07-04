# ⚡ Smart Electricity Consumption Prediction & Energy Optimization System

**TEYZIX CORE Internship — Task ML-3 (Advanced/Industry-Based)**

A production-style Machine Learning system that predicts household daily
electricity consumption, forecasts monthly usage/bills, and generates
personalized energy-saving recommendations — built entirely on an
**original, self-generated synthetic dataset** (no Kaggle/UCI/public data used).

---

## 🎯 Project Objective

Predict daily electricity consumption from household/appliance usage patterns
and provide actionable energy optimization insights via an interactive
Streamlit dashboard.

## 📁 Project Structure

```
Task-3_ElectricityPrediction/
├── data/
│   ├── generate_dataset.py              # Original synthetic dataset generator
│   ├── electricity_dataset.csv          # Raw generated dataset (1212 rows)
│   ├── electricity_dataset_clean.csv    # Cleaned dataset (1200 rows)
│   └── DATASET_DOCUMENTATION.md         # Full dataset documentation
├── src/
│   ├── data_preprocessing.py            # Missing values, duplicates, outliers, encoding
│   ├── eda.py                           # Full EDA + business insights
│   ├── model_training.py                # Feature engineering + 4 models + evaluation
│   └── generate_reports.py              # PDF / Excel / CSV report generation
├── notebooks/
│   └── ML_Pipeline.ipynb                # End-to-end notebook version
├── models/
│   ├── best_model.pkl                   # Best model + scaler + metadata
│   ├── all_models.pkl                   # All 4 trained models
│   └── label_encoders.pkl               # Categorical encoders
├── app/
│   └── streamlit_app.py                 # Prediction interface (kawaii pink theme)
├── reports/
│   ├── eda/                             # All EDA charts (heatmap, histograms, etc.)
│   ├── Project_Report.pdf               # Full PDF project report
│   ├── Energy_Report.xlsx               # Multi-sheet Excel report
│   ├── energy_consumption_report.csv    # Season x House-type consumption pivot
│   ├── model_performance_report.json/csv
│   └── feature_importance.csv
├── requirements.txt
├── Dockerfile                           # Bonus: containerized deployment
└── README.md
```

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt --break-system-packages
```

### 2. Run the full pipeline (in order)
```bash
python3 data/generate_dataset.py        # Step 1: Generate original dataset
python3 src/data_preprocessing.py       # Step 2: Clean data
python3 src/eda.py                      # Step 3: Run EDA
python3 src/model_training.py           # Step 4: Train & evaluate 4 models
python3 src/generate_reports.py         # Step 5: Generate PDF/Excel/CSV reports
```

### 3. Launch the prediction dashboard
```bash
streamlit run app/streamlit_app.py
```

### 4. (Bonus) Run via Docker
```bash
docker build -t electropredict .
docker run -p 8501:8501 electropredict
```

---

## 📊 Dataset

- **1,200 records** (after cleaning) × **19 features + 1 target**
- 100% original — generated via realistic domain-driven synthetic simulation
- Full details in `data/DATASET_DOCUMENTATION.md`

## 🧪 Machine Learning Models Trained & Compared

| Model | MAE | MSE | RMSE | R² Score |
|---|---|---|---|---|
| Linear Regression | 3.916 | 24.472 | 4.947 | 0.7512 |
| Random Forest | 2.055 | 9.287 | 3.047 | 0.9056 |
| **Gradient Boosting (Best)** | **1.905** | **8.295** | **2.880** | **0.9157** |
| XGBoost (Hyperparameter-Tuned) | 1.880 | 8.402 | 2.899 | 0.9146 |

**Best Model:** Gradient Boosting Regressor (R² = 0.9157) — automatically
selected and saved by the training pipeline.

## 🔑 Key Business Insights

- **AC usage hours** is the single strongest driver of consumption
  (correlation ≈ 0.79), followed by number of AC units and outdoor temperature.
- **Summer** is the highest-consuming season (avg ~23 kWh/day) vs **Spring**
  (~12.4 kWh/day, lowest).
- **Bungalows** consume the most on average; **rented portions** consume the least.

## 💡 Prediction System Features (Streamlit App)

- Predicted daily electricity consumption (kWh)
- Estimated monthly electricity usage
- Estimated peak usage hours (heuristic, appliance-load based)
- Estimated monthly bill (progressive slab-rate PKR estimate)
- Energy efficiency score (0–100)
- Personalized, rule-based energy-saving recommendations
- Interactive dashboard tab (season/house-type comparisons)
- **Bonus:** SHAP-based model explainability tab

## 🎁 Bonus Features Implemented

- ✅ Explainable AI (SHAP summary plots in-app)
- ✅ Hyperparameter Optimization (GridSearchCV on XGBoost)
- ✅ Interactive Dashboard (Streamlit, multi-tab)
- ✅ Docker Deployment (Dockerfile included)
- ✅ Electricity Cost Calculator (progressive slab-rate bill estimator)
- ✅ **Live Weather API Integration** (Open-Meteo, free, no API key required —
  enter your city and the outdoor temperature auto-fills from real-time data)
- ✅ Real-Time Prediction (instant in-app inference)
- ✅ What-If Optimization Comparison (unique feature — simulate reduced
  AC/lighting usage and see live savings side-by-side)
- ✅ Appliance-by-Appliance Load Breakdown chart (unique feature)

## ⚠️ Challenges Faced

Designing a synthetic dataset that has **genuine, learnable relationships**
between features and target (rather than pure random noise) required building
an approximate wattage-based load simulation model — balancing realism against
the "no public dataset" constraint.

## 🔮 Future Improvements

- Multi-day / weekly consumption forecasting (time-series extension)
- Real smart-meter data collection to replace/augment synthetic data
- Mobile-responsive UI and push notification alerts for peak usage

## 🛠️ Tech Stack

Python · Pandas · NumPy · Scikit-learn · XGBoost · SHAP · Matplotlib · Seaborn
· Streamlit · ReportLab · OpenPyXL · Docker

---

**Submitted for:** TEYZIX CORE Internship Program — Task ML-3
**Submission Deadline:** 09 July 2026
