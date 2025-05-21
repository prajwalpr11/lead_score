import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from scipy.stats import randint, uniform

from xgboost import XGBClassifier

# 1. Load & rename columns for convenience
df = pd.read_csv('leads.csv')
df = df.rename(columns={
    'Lead Source': 'LeadSource',
    'Total Time Spent on Website': 'TotalTime',
    'Page Views Per Visit':  'PageViews',
    'Last Activity':          'LastActivity'
})

# 2. Feature engineering
df['EngagementScore'] = df['TotalVisits'] * df['PageViews']

# 3. Encode categoricals (for RF and for pipeline’s OneHotEncoder)
le_source   = LabelEncoder().fit(df['LeadSource'])
df['LeadSource_enc']   = le_source.transform(df['LeadSource'])

le_activity = LabelEncoder().fit(df['LastActivity'])
df['LastActivity_enc'] = le_activity.transform(df['LastActivity'])

# 4. Prepare feature matrix & target
feature_cols = [
    'LeadSource_enc',
    'TotalVisits',
    'TotalTime',
    'PageViews',
    'EngagementScore',
    'LastActivity_enc'
]
X = df[feature_cols]
y = df['Converted']

# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 6. Impute any missing (median)
imputer = SimpleImputer(strategy='median')
X_train_imp = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=feature_cols
)
X_test_imp = pd.DataFrame(
    imputer.transform(X_test),
    columns=feature_cols
)

# 7. Fit & calibrate a RandomForest
base_rf    = RandomForestClassifier(n_estimators=100, random_state=42)
calibrator = CalibratedClassifierCV(estimator=base_rf, cv=5)
calibrator.fit(X_train_imp, y_train)
rf_model = calibrator

# 8. Evaluate RF
rf_acc = rf_model.score(X_test_imp, y_test)
print(f"RF Test accuracy with calibration: {rf_acc:.2f}")

# ───── Baseline XGBoost benchmark ─────
# Preprocessor to mirror RF steps
num_features = ['TotalVisits','TotalTime','PageViews','EngagementScore']
cat_features = ['LeadSource_enc','LastActivity_enc']

preprocessor = ColumnTransformer([
    ('num', SimpleImputer(strategy='median'), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),
])

pipe_xgb = Pipeline([
    ('prep', preprocessor),
    ('clf', XGBClassifier(
        eval_metric='auc',
        random_state=42
    ))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
xgb_auc = cross_val_score(
    pipe_xgb, X, y,
    cv=cv, scoring='roc_auc', n_jobs=-1
)
print(f"Baseline XGB CV ROC-AUC: {xgb_auc.mean():.3f} ± {xgb_auc.std():.3f}")

# Fit and get test AUC
pipe_xgb.fit(X_train, y_train)
y_proba = pipe_xgb.predict_proba(X_test_imp)[:,1]
print("Baseline XGB Test ROC-AUC:", roc_auc_score(y_test, y_proba))
# ───────────────────────────────────────

# 9. Hyperparameter tuning for XGBoost
param_dist = {
    'clf__learning_rate':   uniform(0.01, 0.3),
    'clf__n_estimators':    randint(50, 500),
    'clf__max_depth':       randint(3, 10),
    'clf__subsample':       uniform(0.6, 0.4),
    'clf__colsample_bytree':uniform(0.6, 0.4),
}

search_xgb = RandomizedSearchCV(
    pipe_xgb,
    param_dist,
    n_iter=30,
    cv=cv,
    scoring='roc_auc',
    random_state=42,
    n_jobs=-1,
    refit=True
)
search_xgb.fit(X_train, y_train)

print("🔍 Best XGB CV AUC:", search_xgb.best_score_)
print("🔧 Best XGB params:", search_xgb.best_params_)

# 10. Evaluate tuned XGB on held-out test set
best_xgb = search_xgb.best_estimator_
y_tuned_proba = best_xgb.predict_proba(X_test_imp)[:,1]
print("Tuned XGB Test ROC-AUC:", roc_auc_score(y_test, y_tuned_proba))

# 11. Save models & preprocessing artifacts
joblib.dump(rf_model,        'model_calibrated.pkl')
joblib.dump(imputer,         'imputer.pkl')
joblib.dump(le_source,       'le_source.pkl')
joblib.dump(le_activity,     'le_activity.pkl')
joblib.dump(best_xgb,        'model_xgb.pkl')
print("✅ All artifacts saved: RF, encoders, imputer, and tuned XGB pipeline.")
