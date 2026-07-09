import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Load the cleaned data
df = pd.read_csv('data/cleaned_data_final.csv')

# Identify categorical columns
categorical_cols = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_INCOME_TYPE', 
                   'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 
                   'OCCUPATION_TYPE', 'FLAG_MOBIL', 'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL']

# Encode categorical variables
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le
    print(f"Encoded {col}")

# Select features for modeling
features = ['CODE_GENDER_encoded', 'FLAG_OWN_CAR_encoded', 'FLAG_OWN_REALTY_encoded', 
           'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'NAME_INCOME_TYPE_encoded', 
           'NAME_EDUCATION_TYPE_encoded', 'NAME_FAMILY_STATUS_encoded', 
           'NAME_HOUSING_TYPE_encoded', 'AGE', 'EMPLOYMENT_YEARS', 
           'FLAG_MOBIL_encoded', 'FLAG_WORK_PHONE_encoded', 'FLAG_PHONE_encoded', 
           'FLAG_EMAIL_encoded', 'OCCUPATION_TYPE_encoded', 'CNT_FAM_MEMBERS']

X = df[features]
y = df['TARGET'].map({'good': 0, 'bad': 1})  # 0=good, 1=bad

print(f"\nFeatures shape: {X.shape}")
print(f"Target distribution:")
print(y.value_counts())

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"\nTraining target distribution:")
print(y_train.value_counts())
print(f"\nTest target distribution:")
print(y_test.value_counts())

# Train a Logistic Regression model (baseline)
print("\n=== TRAINING LOGISTIC REGRESSION ===")
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)

# Make predictions
y_pred_log = log_reg.predict(X_test)
y_pred_proba_log = log_reg.predict_proba(X_test)[:, 1]

# Evaluate
print("\nClassification Report:")
print(classification_report(y_test, y_pred_log))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba_log):.4f}")

from sklearn.preprocessing import StandardScaler

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features scaled successfully!")

# Train Logistic Regression with class weight to handle imbalance
print("\n=== TRAINING LOGISTIC REGRESSION WITH CLASS WEIGHT ===")
log_reg_weighted = LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced')
log_reg_weighted.fit(X_train_scaled, y_train)

y_pred_weighted = log_reg_weighted.predict(X_test_scaled)
y_pred_proba_weighted = log_reg_weighted.predict_proba(X_test_scaled)[:, 1]

print("\nClassification Report (with class weights):")
print(classification_report(y_test, y_pred_weighted))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba_weighted):.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'coefficient': log_reg_weighted.coef_[0]
}).sort_values('coefficient', key=abs, ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))
# Train Random Forest with class weight
print("\n=== TRAINING RANDOM FOREST ===")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)

y_pred_rf = rf_model.predict(X_test_scaled)
y_pred_proba_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

print("\nClassification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba_rf):.4f}")

# Feature importance from Random Forest
feature_importance_rf = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features (Random Forest):")
print(feature_importance_rf.head(10))
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

# Apply SMOTE to handle class imbalance
print("\n=== APPLYING SMOTE ===")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

print(f"Original training set: {X_train_scaled.shape[0]} samples")
print(f"After SMOTE: {X_train_smote.shape[0]} samples")
print(f"Class distribution after SMOTE:")
print(pd.Series(y_train_smote).value_counts())

# Train Random Forest on SMOTE data
print("\n=== TRAINING RANDOM FOREST WITH SMOTE ===")
rf_smote = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf_smote.fit(X_train_smote, y_train_smote)

y_pred_rf_smote = rf_smote.predict(X_test_scaled)
y_pred_proba_rf_smote = rf_smote.predict_proba(X_test_scaled)[:, 1]

print("\nClassification Report (Random Forest + SMOTE):")
print(classification_report(y_test, y_pred_rf_smote))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba_rf_smote):.4f}")
import xgboost as xgb

print("\n=== TRAINING XGBOOST ===")
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=7.5,  # ratio of good/bad to handle imbalance
    random_state=42,
    eval_metric='logloss'
)
xgb_model.fit(X_train_scaled, y_train)

y_pred_xgb = xgb_model.predict(X_test_scaled)
y_pred_proba_xgb = xgb_model.predict_proba(X_test_scaled)[:, 1]

print("\nClassification Report (XGBoost):")
print(classification_report(y_test, y_pred_xgb))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba_xgb):.4f}")

# Feature importance from XGBoost
feature_importance_xgb = pd.DataFrame({
    'feature': features,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features (XGBoost):")
print(feature_importance_xgb.head(10))
from sklearn.model_selection import GridSearchCV

print("\n=== HYPERPARAMETER TUNING FOR XGBOOST ===")

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1],
    'scale_pos_weight': [5, 7.5, 10]
}

# Create XGBoost classifier
xgb_tune = xgb.XGBClassifier(random_state=42, eval_metric='logloss')

# Grid search with 3-fold cross validation
grid_search = GridSearchCV(
    xgb_tune, 
    param_grid, 
    cv=3, 
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best cross-validation ROC-AUC: {grid_search.best_score_:.4f}")

# Train model with best parameters
best_xgb = grid_search.best_estimator_
y_pred_best = best_xgb.predict(X_test_scaled)
y_pred_proba_best = best_xgb.predict_proba(X_test_scaled)[:, 1]

print("\nClassification Report (Tuned XGBoost):")
print(classification_report(y_test, y_pred_best))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba_best):.4f}")
import joblib

# Save the best model, scaler, and label encoders
print("\n=== SAVING MODEL AND PREPROCESSORS ===")

# Save model
joblib.dump(best_xgb, 'models/best_xgb_model.pkl')
print("✅ Model saved to 'models/best_xgb_model.pkl'")

# Save scaler
joblib.dump(scaler, 'models/scaler.pkl')
print("✅ Scaler saved to 'models/scaler.pkl'")

# Save label encoders
joblib.dump(le_dict, 'models/label_encoders.pkl')
print("✅ Label encoders saved to 'models/label_encoders.pkl'")

# Save feature names
joblib.dump(features, 'models/features.pkl')
print("✅ Feature names saved to 'models/features.pkl'")

print("\n✅ All artifacts saved successfully!")