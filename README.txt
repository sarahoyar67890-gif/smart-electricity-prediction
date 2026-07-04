TEYZIX CORE INTERNSHIP - TASK 3 SUBMISSION
============================================
Task ID: ML-3
Title: Smart Electricity Consumption Prediction & Energy Optimization System
Domain: Machine Learning

SHORT DESCRIPTION
------------------
A complete, production-style ML system that predicts household daily
electricity consumption from appliance usage patterns using an ORIGINAL,
self-generated synthetic dataset (1,200 records, 19 features). The system
compares 4 regression algorithms (Linear Regression, Random Forest, Gradient
Boosting, XGBoost), selects the best performer (Gradient Boosting, R2=0.9157),
and serves predictions through a kawaii-pink-themed interactive Streamlit
dashboard that also provides monthly bill estimates, peak usage hours, an
efficiency score, and personalized energy-saving recommendations.

HOW TO RUN
-----------
1. pip install -r requirements.txt --break-system-packages
2. python3 data/generate_dataset.py
3. python3 src/data_preprocessing.py
4. python3 src/eda.py
5. python3 src/model_training.py
6. python3 src/generate_reports.py
7. streamlit run app/streamlit_app.py

   (Optional) Docker: docker build -t electropredict . && docker run -p 8501:8501 electropredict

WHAT'S INCLUDED
-----------------
- Original dataset + generator script + full dataset documentation
- Complete preprocessing, EDA, feature engineering pipeline
- 4 trained ML models with full evaluation (MAE/MSE/RMSE/R2)
- Jupyter Notebook (fully executed, no errors) covering the whole lifecycle
- Streamlit prediction app with all required outputs + SHAP explainability
- PDF, Excel, and CSV reports (dataset summary, model performance, forecasts,
  recommendations)
- README.md with full project documentation
- Dockerfile for containerized deployment (bonus)

See README.md for full details.
