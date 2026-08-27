# Task 4: Customer Churn Prediction

## 📌 Project Overview
This project builds a machine learning model to predict whether a telecom customer is likely to churn (leave the company), based on their account details, services used, and billing information. This task was completed as part of the Data Science Internship at SAM AI Technologies.

## 📊 Dataset
- **Source:** [Telco Customer Churn - Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **File Used:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Rows:** 7043
- **Features:** 20 attributes covering demographics (gender, senior citizen, partner, dependents), account info (tenure, contract, payment method, charges), and services subscribed (phone, internet, streaming, tech support, etc.)
- **Target:** `Churn` — Yes (customer left) / No (customer stayed)

## 🛠️ Tools & Libraries
- Python
- Pandas & NumPy (data handling)
- Matplotlib & Seaborn (visualization)
- Scikit-learn (machine learning models & evaluation)

## 🔍 Workflow
1. **Data Loading & Exploration** — checked shape, data types, and dataset structure (7043 rows, 21 columns)
2. **Feature Engineering & Cleaning** —
   - Dropped `customerID` (not predictive)
   - Fixed `TotalCharges`, which was incorrectly stored as text; converted to numeric and filled 11 resulting missing values with the median
   - Encoded the target column: Yes → 1, No → 0
   - Encoded all other categorical (text) columns using Label Encoding
3. **Train/Test Split** — 80/20 split with stratification to preserve the class balance (5,174 stayed vs 1,869 churned)
4. **Model Training** — trained and compared three classification models:
   - Logistic Regression
   - Decision Tree Classifier
   - Random Forest Classifier
5. **Evaluation** — compared models using Accuracy, Precision, Recall, F1-Score, and **ROC-AUC**
6. **Visualization** — plotted ROC curves for all models, a Confusion Matrix for the best model, and a Feature Importance chart

## 📈 Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.799 | 0.643 | 0.548 | 0.592 | **0.840** |
| Decision Tree | 0.730 | 0.492 | 0.519 | 0.505 | 0.663 |
| Random Forest | 0.792 | 0.637 | 0.503 | 0.562 | 0.823 |

**Best Model:** Logistic Regression (ROC-AUC: 0.840)

### Confusion Matrix (Logistic Regression)
- **True Negatives (correctly predicted "Stayed"):** 921
- **False Positives (predicted "Churned" but actually stayed):** 114
- **False Negatives (predicted "Stayed" but actually churned):** 169
- **True Positives (correctly predicted "Churned"):** 205

### Key Insights
- **TotalCharges**, **MonthlyCharges**, and **tenure** were the three most important features for predicting churn, followed by **Contract type**.
- An ROC-AUC of 0.84 shows the model is quite good at distinguishing between customers who will churn and those who won't — well above the 0.5 baseline of random guessing.
- Logistic Regression outperformed the tree-based models on ROC-AUC, suggesting churn risk correlates fairly smoothly with billing amount and tenure rather than requiring complex non-linear splits.
- Recall for the churn class (55%) is noticeably lower than for the "stayed" class, meaning the model misses a meaningful portion of customers who actually churn (169 false negatives). In a real business setting, this is an important trade-off to consider — improving recall (e.g., via class weighting or threshold tuning) could help catch more at-risk customers, at the cost of more false alarms.
- This class imbalance (5,174 vs 1,869) is a likely factor behind the gap between precision and recall across all models.

## 📁 Files in this Folder
- `customer_churn_prediction.py` — Full source code (data loading, cleaning, feature engineering, model training, evaluation)
- `README.md` — Project documentation (this file)

## ✅ How to Run
1. Download the dataset from the Kaggle link above (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)
2. Open the script in Google Colab or Jupyter Notebook
3. Upload the CSV file to the same environment
4. Run all cells to reproduce the results

---
*Submitted as part of the Data Science Internship — SAM AI Technologies*
