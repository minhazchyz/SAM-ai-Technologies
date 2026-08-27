# Task 3: Loan Approval Prediction

## 📌 Project Overview
This project builds a classification model to predict whether a loan application will be approved, based on applicant details such as income, credit history, and property area. This task was completed as part of the Data Science Internship at SAM AI Technologies.

## 📊 Dataset
- **Source:** [Loan Prediction Problem Dataset - Kaggle](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)
- **File Used:** `train.csv`
- **Rows:** 614
- **Features:** Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area
- **Target:** `Loan_Status` — Y (Approved) / N (Not Approved)

## 🛠️ Tools & Libraries
- Python
- Pandas & NumPy (data handling)
- Matplotlib & Seaborn (visualization)
- Scikit-learn (machine learning models & evaluation)

## 🔍 Workflow
1. **Data Loading & Exploration** — checked shape, data types, and missing values
2. **Handling Missing Values** — the dataset had missing values in several columns (Gender, Married, Dependents, Self_Employed, LoanAmount, Loan_Amount_Term, Credit_History). Numeric columns were filled with the **median**, categorical columns were filled with the **mode** (most frequent value)
3. **Encoding** — categorical features were converted to numeric using Label Encoding; target column encoded as Y → 1, N → 0
4. **Train/Test Split** — 80/20 split with stratification to preserve class balance
5. **Model Training** — trained and compared three classification models:
   - Logistic Regression
   - Decision Tree Classifier
   - Random Forest Classifier
6. **Evaluation** — compared models using Accuracy, Precision, Recall, and F1-Score
7. **Visualization** — plotted a Confusion Matrix and Feature Importance chart for the best model

## 📈 Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 0.862 | 0.840 | 0.988 | **0.908** |
| Decision Tree | 0.764 | 0.826 | 0.835 | 0.830 |
| Random Forest | 0.829 | 0.848 | 0.918 | 0.881 |

**Best Model:** Logistic Regression (F1-Score: 0.908)

### Confusion Matrix (Logistic Regression)
- **True Negatives (correctly predicted "Not Approved"):** 22
- **False Positives (predicted "Approved" but actually not):** 16
- **False Negatives (predicted "Not Approved" but actually approved):** 1
- **True Positives (correctly predicted "Approved"):** 84

### Key Insights
- **Credit_History** was by far the most important feature in predicting loan approval, followed by **ApplicantIncome**, **LoanAmount**, and **CoapplicantIncome**.
- The model is very good at correctly identifying approved loans (recall of 0.99 for the "Approved" class), but has more difficulty correctly flagging rejected loans (only 58% recall for the "Not Approved" class — 22 out of 38 actual rejections were correctly identified).
- This pattern is common in loan datasets where approvals significantly outnumber rejections (422 vs 192 in this dataset), causing the model to lean toward predicting approval.
- Logistic Regression outperformed the more complex tree-based models here, suggesting the relationship between features (especially Credit_History) and loan approval is largely linear/threshold-based.

## 📁 Files in this Folder
- `loan_approval_prediction.py` — Full source code (data loading, missing value handling, encoding, model training, evaluation)
- `README.md` — Project documentation (this file)

## ✅ How to Run
1. Download the dataset from the Kaggle link above and extract `train.csv`
2. Open the script in Google Colab or Jupyter Notebook
3. Upload `train.csv` to the same environment
4. Run all cells to reproduce the results

---
*Submitted as part of the Data Science Internship — SAM AI Technologies*
