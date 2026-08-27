# Task 1: Heart Disease Prediction

## 📌 Project Overview
This project builds a machine learning classification model to predict whether a patient has heart disease based on various medical attributes. This task was completed as part of the Data Science Internship at SAM AI Technologies.

## 📊 Dataset
- **Source:** [Heart Disease Dataset - Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- **Rows:** 1025
- **Features:** 13 medical attributes (age, sex, chest pain type, resting blood pressure, cholesterol, etc.)
- **Target:** Binary classification — 1 (Heart Disease) / 0 (No Disease)

## 🛠️ Tools & Libraries
- Python
- Pandas & NumPy (data handling)
- Matplotlib & Seaborn (visualization)
- Scikit-learn (machine learning models & evaluation)

## 🔍 Workflow
1. **Data Loading & Exploration** — checked shape, data types, and missing values
2. **Preprocessing** — split features/target, applied train-test split (80/20), scaled features for Logistic Regression
3. **Model Training** — trained and compared three models:
   - Logistic Regression
   - Decision Tree Classifier
   - Random Forest Classifier
4. **Evaluation** — compared models using Accuracy, Precision, Recall, and F1-Score
5. **Visualization** — plotted a Confusion Matrix and Feature Importance chart for the best model

## 📈 Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 0.795 | 0.756 | 0.874 | 0.811 |
| Decision Tree | 0.985 | 1.000 | 0.971 | 0.985 |
| Random Forest | 0.985 | 1.000 | 0.971 | 0.985 |

**Best Model:** Decision Tree / Random Forest (F1-Score: 0.985)

### Key Insights
- The most important features for predicting heart disease were **cp (chest pain type)**, **ca**, **thalach (max heart rate)**, and **oldpeak**.
- The Confusion Matrix shows only 3 misclassifications out of 205 test samples.

## 📁 Files in this Folder
- `heart_disease_prediction.py` — Full source code (data loading, preprocessing, model training, evaluation)
- `README.md` — Project documentation (this file)

## ✅ How to Run
1. Download the dataset from the Kaggle link above (`heart.csv`)
2. Open the script in Google Colab or Jupyter Notebook
3. Upload `heart.csv` to the same environment
4. Run all cells to reproduce the results

---
*Submitted as part of the Data Science Internship — SAM AI Technologies*
