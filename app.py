from flask import Flask, request, jsonify
import pandas as pd
import joblib

# ─── Load the tuned XGB pipeline and your encoders ───
model        = joblib.load('model_xgb.pkl')      # your full Pipeline from train_model.py
le_source    = joblib.load('le_source.pkl')      # LabelEncoder for LeadSource
le_activity  = joblib.load('le_activity.pkl')    # LabelEncoder for LastActivity
# ─────────────────────────────────────────────────────

app = Flask(__name__)

@app.route('/score', methods=['POST'])
def score():
    data = request.json

    # Build feature DataFrame
    df = pd.DataFrame([{
        'LeadSource':   data['LeadSource'],
        'TotalVisits':  data['TotalVisits'],
        'TotalTime':    data['TotalTime'],
        'PageViews':    data['PageViews'],
        'LastActivity': data['LastActivity']
    }])

    # 1. Feature engineering
    df['EngagementScore'] = df['TotalVisits'] * df['PageViews']

    # 2. Encode categorical inputs
    df['LeadSource_enc']   = le_source.transform(df['LeadSource'])
    df['LastActivity_enc'] = le_activity.transform(df['LastActivity'])

    # 3. Select feature columns in the same order your pipeline expects
    feature_cols = [
        'LeadSource_enc',
        'TotalVisits',
        'TotalTime',
        'PageViews',
        'EngagementScore',
        'LastActivity_enc'
    ]
    X = df[feature_cols]

    # 4. Predict probability & scale to 0–100
    proba = model.predict_proba(X)[0][1]
    score = int(round(proba * 100))

    return jsonify({'score': score})

if __name__ == '__main__':
    app.run(debug=True)
