import joblib
import pandas as pd
import numpy as np

# Load all saved artifacts
print("Loading model artifacts...")
model = joblib.load('models/best_xgb_model.pkl')
scaler = joblib.load('models/scaler.pkl')
le_dict = joblib.load('models/label_encoders.pkl')
features = joblib.load('models/features.pkl')

print("✅ All artifacts loaded successfully!")

def predict_client(input_data):
    """
    Predict if a client is 'good' or 'bad'
    input_data: dict with client information
    """
    # Create DataFrame from input
    df = pd.DataFrame([input_data])
    
    # Encode categorical variables
    for col in le_dict:
        if col in df.columns:
            try:
                df[col + '_encoded'] = le_dict[col].transform(df[col].astype(str))
            except ValueError:
                # Handle unknown categories
                df[col + '_encoded'] = -1
    
    # Select features in correct order
    X = df[features]
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make prediction
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0][1]
    
    result = {
        'prediction': 'bad' if prediction == 1 else 'good',
        'probability_bad': probability,
        'probability_good': 1 - probability
    }
    
    return result

# Test with sample data
if __name__ == "__main__":
    # Test with a high-risk client
    high_risk_client = {
        'CODE_GENDER': 'M',
        'FLAG_OWN_CAR': 'N',
        'FLAG_OWN_REALTY': 'N',
        'CNT_CHILDREN': 3,
        'AMT_INCOME_TOTAL': 45000,
        'NAME_INCOME_TYPE': 'Unemployed',
        'NAME_EDUCATION_TYPE': 'Secondary / secondary special',
        'NAME_FAMILY_STATUS': 'Single / not married',
        'NAME_HOUSING_TYPE': 'Rented apartment',
        'AGE': 25,
        'EMPLOYMENT_YEARS': 0,
        'FLAG_MOBIL': 1,
        'FLAG_WORK_PHONE': 0,
        'FLAG_PHONE': 0,
        'FLAG_EMAIL': 0,
        'OCCUPATION_TYPE': 'Unknown',
        'CNT_FAM_MEMBERS': 4
    }

    result_high = predict_client(high_risk_client)
    print("\n=== HIGH-RISK CLIENT PREDICTION ===")
    print(f"Prediction: {result_high['prediction']}")
    print(f"Probability of being BAD: {result_high['probability_bad']:.2%}")
    print(f"Probability of being GOOD: {result_high['probability_good']:.2%}")

    # Test with a good client
    sample_client = {
        'CODE_GENDER': 'M',
        'FLAG_OWN_CAR': 'Y',
        'FLAG_OWN_REALTY': 'Y',
        'CNT_CHILDREN': 0,
        'AMT_INCOME_TOTAL': 270000,
        'NAME_INCOME_TYPE': 'Working',
        'NAME_EDUCATION_TYPE': 'Higher education',
        'NAME_FAMILY_STATUS': 'Married',
        'NAME_HOUSING_TYPE': 'House / apartment',
        'AGE': 35,
        'EMPLOYMENT_YEARS': 5,
        'FLAG_MOBIL': 1,
        'FLAG_WORK_PHONE': 0,
        'FLAG_PHONE': 0,
        'FLAG_EMAIL': 1,
        'OCCUPATION_TYPE': 'Sales staff',
        'CNT_FAM_MEMBERS': 2
    }
    
    result = predict_client(sample_client)
    print("\n=== GOOD CLIENT PREDICTION ===")
    print(f"Prediction: {result['prediction']}")
    print(f"Probability of being BAD: {result['probability_bad']:.2%}")
    print(f"Probability of being GOOD: {result['probability_good']:.2%}")

    # Load test data to find actual bad clients
df_test = pd.read_csv('data/cleaned_data_final.csv')

# Get a few actual bad clients
bad_clients = df_test[df_test['TARGET'] == 'bad'].head(5)

print("\n=== TESTING ON ACTUAL BAD CLIENTS FROM DATASET ===")
for idx, row in bad_clients.iterrows():
    # Prepare input data
    client_data = {
        'CODE_GENDER': row['CODE_GENDER'],
        'FLAG_OWN_CAR': row['FLAG_OWN_CAR'],
        'FLAG_OWN_REALTY': row['FLAG_OWN_REALTY'],
        'CNT_CHILDREN': row['CNT_CHILDREN'],
        'AMT_INCOME_TOTAL': row['AMT_INCOME_TOTAL'],
        'NAME_INCOME_TYPE': row['NAME_INCOME_TYPE'],
        'NAME_EDUCATION_TYPE': row['NAME_EDUCATION_TYPE'],
        'NAME_FAMILY_STATUS': row['NAME_FAMILY_STATUS'],
        'NAME_HOUSING_TYPE': row['NAME_HOUSING_TYPE'],
        'AGE': row['AGE'],
        'EMPLOYMENT_YEARS': row['EMPLOYMENT_YEARS'],
        'FLAG_MOBIL': row['FLAG_MOBIL'],
        'FLAG_WORK_PHONE': row['FLAG_WORK_PHONE'],
        'FLAG_PHONE': row['FLAG_PHONE'],
        'FLAG_EMAIL': row['FLAG_EMAIL'],
        'OCCUPATION_TYPE': row['OCCUPATION_TYPE'] if pd.notna(row['OCCUPATION_TYPE']) else 'Unknown',
        'CNT_FAM_MEMBERS': row['CNT_FAM_MEMBERS']
    }
    
    result = predict_client(client_data)
    print(f"\nClient ID: {row['ID']}")
    print(f"Actual: bad | Predicted: {result['prediction']}")
    print(f"Probability BAD: {result['probability_bad']:.2%}")