LeadScoring — AI-Powered Lead Prioritization (Phase 1)

Purpose: small end-to-end demo: train a model on sample leads (MathWorks online courses target), expose a scoring REST API (Flask), call it from Salesforce via an Apex callout, and show score in an LWC on Lead pages.

Contents:

train_model.py — trains XGBoost & RandomForest baseline, saves artifacts (pipelines, encoders, imputer, model).

app.py — Flask app that loads the saved artifacts and exposes POST /score.

Model artifacts (saved by joblib.dump): model_xgb.pkl, imputer.pkl, le_source.pkl, le_activity.pkl, model_calibrated.pkl (names may vary per your code).

leads.csv — sample dataset used for training.

salesforce/ — folder with Apex and LWC source (example files).

requirements.txt — Python packages for Render / local environment.

Quick Start — Local (development)

Create & activate a Python environment (recommended):

python -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows
pip install -r requirements.txt


Train the model artifacts:

python train_model.py
# This should create files like model_xgb.pkl, imputer.pkl, le_source.pkl, le_activity.pkl, model_calibrated.pkl


Run the Flask app locally:

# development (auto-reloads)
python app.py
# or for production-like serving
pip install gunicorn
gunicorn "app:app" --bind 0.0.0.0:5000


Test with curl:

curl -X POST http://127.0.0.1:5000/score \
  -H "Content-Type: application/json" \
  -d '{
    "LeadSource":"Organic Search",
    "TotalVisits":5,
    "TotalTime":674,
    "PageViews":2.5,
    "LastActivity":"Email Opened"
  }'
# Expect: {"score": <0-100>}

Deploying to Render (or similar services)

Render has a free tier suitable for demos (with some limitations like sleeping services on free tier). Steps:

In Render dashboard:

Create a new Web Service.

Connect your GitHub repo.

Build command:

pip install -r requirements.txt


Start command (use gunicorn for production):

gunicorn "app:app" --bind 0.0.0.0:$PORT


Environment: Render sets $PORT, but no extra env vars are required if your API is public.

Ensure requirements.txt contains necessary packages:

flask
pandas
numpy
scikit-learn
xgboost
joblib
gunicorn
scipy


Deploy and copy the public URL (e.g. https://lead-scorer-api.onrender.com).
