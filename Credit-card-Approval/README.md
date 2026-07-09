
# 💳 Credit Card Approval Predictor

A machine learning project that predicts whether a credit card applicant is a **"good"** or **"bad"** client using historical data. Built with **XGBoost**, **Random Forest**, and deployed with an interactive **Flask** web UI.

---

## 📌 Project Overview

This project uses a real-world credit card dataset to assess the risk of a loan applicant defaulting. The goal is to help financial institutions make informed decisions by predicting the probability of a client being "bad" (likely to default) or "good" (likely to repay).

The dataset contains:
- **438,557** applicant records
- **1,048,575** credit history records
- **18** features including income, age, employment, family status, and more

---

## 🧠 Models Used

| Model | ROC-AUC Score | Recall (Bad) | Precision (Bad) |
|-------|---------------|--------------|-----------------|
| Logistic Regression | 0.5288 | 0.00 | 0.00 |
| Logistic + Class Weight | 0.5455 | 0.50 | 0.13 |
| **Random Forest** | **0.7205** | 0.51 | 0.26 |
| XGBoost | 0.6948 | 0.54 | 0.22 |
| **Tuned XGBoost** ✅ | **0.7376** | **0.69** | **0.22** |

### 🏆 Best Model: Tuned XGBoost

After hyperparameter tuning using **GridSearchCV**, XGBoost achieved the best performance:

- **ROC-AUC Score:** `0.7376`
- **Recall for Bad Class:** `0.69` (catches 69% of bad clients)
- **Precision for Bad Class:** `0.22`

---

## 🔧 Features Used

The model uses the following features:

- `CODE_GENDER` - Gender
- `FLAG_OWN_CAR` - Owns a car
- `FLAG_OWN_REALTY` - Owns property
- `CNT_CHILDREN` - Number of children
- `AMT_INCOME_TOTAL` - Annual income
- `NAME_INCOME_TYPE` - Income category
- `NAME_EDUCATION_TYPE` - Education level
- `NAME_FAMILY_STATUS` - Marital status
- `NAME_HOUSING_TYPE` - Housing type
- `AGE` - Age (derived from `DAYS_BIRTH`)
- `EMPLOYMENT_YEARS` - Years employed (derived from `DAYS_EMPLOYED`)
- `OCCUPATION_TYPE` - Occupation
- `CNT_FAM_MEMBERS` - Family size

---

## 📊 Dataset Information

- **Source:** Credit Card Dataset for Machine Learning
- **Target:** `good` (0) or `bad` (1)
- **Class Distribution:**
  - `good`: 32,166 (88.23%)
  - `bad`: 4,291 (11.77%)
- **Imbalance Ratio:** 7.5:1

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Siddique-ur-Rehman/ML-Projects.git
cd ML-Projects/Credit-card-Approval
```

### 2️⃣ Create a Virtual Environment

Using `uv` (recommended) or `venv`:

```bash
# Using uv
uv venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Mac/Linux)
source .venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
uv pip install -r requirements.txt
```

### 4️⃣ Run the Flask Web App

```bash
python app.py
```

Open your browser and go to: **http://127.0.0.1:5000**

---

## 🖥️ Web Interface

The Flask UI allows you to:

1. Enter applicant details via a clean web form
2. Get instant predictions with probability scores
3. View a summary of applicant information

---

## 📁 Project Structure

```
Credit-card-Approval/
├── data/
│   ├── application_record.csv
│   ├── credit_record.csv
│   ├── cleaned_data.csv
│   └── cleaned_data_final.csv
├── models/
│   ├── best_xgb_model.pkl
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   └── features.pkl
├── templates/
│   ├── index.html
│   └── result.html
├── app.py                  # Flask web application
├── predict.py              # Prediction script
├── model_training.py       # Training pipeline
├── data_exploration.py     # Data analysis
├── requirements.txt        # Dependencies
└── README.md               # Project documentation
```

---

## 🛠️ Technologies Used

| Tool | Purpose |
|------|---------|
| **Python** | Main programming language |
| **Pandas / NumPy** | Data manipulation |
| **Scikit-learn** | Preprocessing, Logistic Regression, Random Forest |
| **XGBoost** | Gradient boosting model |
| **Imbalanced-learn** | SMOTE for handling imbalance |
| **Matplotlib / Seaborn** | Data visualization |
| **Flask** | Web framework |
| **HTML / CSS** | Frontend UI |

---

## 📈 Results & Insights

### Key Findings

- **Age** is the most important feature for predicting risk
- **Income** and **Employment Years** are also strong predictors
- Class imbalance is a major challenge (7.5:1)
- XGBoost with class weighting outperforms other models

### Feature Importance (XGBoost)

| Feature | Importance |
|---------|------------|
| AGE | 17.77% |
| AMT_INCOME_TOTAL | 15.75% |
| EMPLOYMENT_YEARS | 12.24% |
| OCCUPATION_TYPE | 10.84% |

---

## 🔗 Links

- 📂 [GitHub Repository](https://github.com/Siddique-ur-Rehman/ML-Projects)
- 💳 [Credit Card Dataset](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction)
- 📚 [XGBoost Documentation](https://xgboost.readthedocs.io/)
- 🌐 [Flask Documentation](https://flask.palletsprojects.com/)

---

## 🧪 Future Improvements

- [ ] Deploy to cloud (AWS, Heroku, or Render)
- [ ] Add more advanced features (credit utilization, payment history)
- [ ] Use deep learning (Neural Networks)
- [ ] Add model explainability with SHAP or LIME
- [ ] Real-time API integration

---

## 👨‍💻 Author

**Siddique-ur-Rehman**

- GitHub: [@Siddique-ur-Rehman](https://github.com/Siddique-ur-Rehman)
- Email: siddique5623530@gmail.com

---



