from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model artifacts
print("Loading model artifacts...")
model = joblib.load('models/best_xgb_model.pkl')
scaler = joblib.load('models/scaler.pkl')
le_dict = joblib.load('models/label_encoders.pkl')
features = joblib.load('models/features.pkl')
print("✅ All artifacts loaded successfully!")

def predict_client(input_data):
    """Predict if a client is 'good' or 'bad'"""
    df = pd.DataFrame([input_data])
    
    # Encode categorical variables
    for col in le_dict:
        if col in df.columns:
            try:
                df[col + '_encoded'] = le_dict[col].transform(df[col].astype(str))
            except ValueError:
                df[col + '_encoded'] = -1
    
    # Select features and scale
    X = df[features]
    X_scaled = scaler.transform(X)
    
    # Make prediction
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0][1]
    
    return {
        'prediction': 'bad' if prediction == 1 else 'good',
        'probability_bad': round(probability * 100, 2),
        'probability_good': round((1 - probability) * 100, 2)
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = {
            'CODE_GENDER': request.form['CODE_GENDER'],
            'FLAG_OWN_CAR': request.form['FLAG_OWN_CAR'],
            'FLAG_OWN_REALTY': request.form['FLAG_OWN_REALTY'],
            'CNT_CHILDREN': int(request.form['CNT_CHILDREN']),
            'AMT_INCOME_TOTAL': float(request.form['AMT_INCOME_TOTAL']),
            'NAME_INCOME_TYPE': request.form['NAME_INCOME_TYPE'],
            'NAME_EDUCATION_TYPE': request.form['NAME_EDUCATION_TYPE'],
            'NAME_FAMILY_STATUS': request.form['NAME_FAMILY_STATUS'],
            'NAME_HOUSING_TYPE': request.form['NAME_HOUSING_TYPE'],
            'AGE': int(request.form['AGE']),
            'EMPLOYMENT_YEARS': int(request.form['EMPLOYMENT_YEARS']),
            'FLAG_MOBIL': int(request.form['FLAG_MOBIL']),
            'FLAG_WORK_PHONE': int(request.form['FLAG_WORK_PHONE']),
            'FLAG_PHONE': int(request.form['FLAG_PHONE']),
            'FLAG_EMAIL': int(request.form['FLAG_EMAIL']),
            'OCCUPATION_TYPE': request.form['OCCUPATION_TYPE'],
            'CNT_FAM_MEMBERS': int(request.form['CNT_FAM_MEMBERS'])
        }
        
        result = predict_client(data)
        return render_template('result.html', result=result, data=data)
    
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)