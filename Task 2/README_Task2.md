# Task 2: Student Performance Prediction

## 📌 Project Overview
This project analyzes factors affecting student academic performance and builds regression models to predict a student's average exam score based on demographic and background attributes. This task was completed as part of the Data Science Internship at SAM AI Technologies.

## 📊 Dataset
- **Source:** [Students Performance in Exams - Kaggle](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)
- **Rows:** 1000
- **Original Features:** gender, race/ethnicity, parental level of education, lunch type, test preparation course, math score, reading score, writing score
- **Target:** `average_score` — created by averaging math, reading, and writing scores

## 🛠️ Tools & Libraries
- Python
- Pandas & NumPy (data handling)
- Matplotlib & Seaborn (visualization)
- Scikit-learn (machine learning models & evaluation)

## 🔍 Workflow
1. **Data Loading & Exploration** — checked shape, data types, and missing values (no missing values found)
2. **Feature Engineering** — created `average_score` from math, reading, and writing scores
3. **Preprocessing** — encoded categorical columns (gender, race/ethnicity, parental education, lunch, test prep) using Label Encoding, applied an 80/20 train-test split
4. **Model Training** — trained and compared three regression models using only demographic/background features (math, reading, and writing scores were excluded from the input features to avoid data leakage):
   - Linear Regression
   - Decision Tree Regressor
   - Random Forest Regressor
5. **Evaluation** — compared models using MAE, RMSE, and R2 Score
6. **Visualization** — plotted Actual vs Predicted scores and a Feature Importance chart

## 📈 Results

| Model | MAE | RMSE | R2 Score |
|---|---|---|---|
| Linear Regression | 10.72 | 13.69 | **0.126** |
| Decision Tree | 11.92 | 15.31 | -0.093 |
| Random Forest | 11.54 | 14.85 | -0.029 |

**Best Model:** Linear Regression (R2 Score: 0.126)

### Key Insights
- **Parental level of education** and **race/ethnicity** were the two most influential features on predicted student performance, followed by lunch type and test preparation course.
- The R2 scores are relatively low across all models, which shows that demographic and background factors alone are only weakly predictive of exam performance — a student's actual scores are influenced by many other factors (study habits, individual ability, attendance, etc.) not captured in this dataset.
- Tree-based models (Decision Tree, Random Forest) performed worse than Linear Regression here, likely because the categorical features don't have strong non-linear interactions to exploit, and the small feature set makes tree models prone to overfitting/noise.
- This is a useful and realistic finding: it highlights that academic performance can't be reliably predicted from demographics alone, which is itself a meaningful conclusion for this kind of analysis.

## 📁 Files in this Folder
- `student_performance_prediction.py` — Full source code (data loading, preprocessing, model training, evaluation)
- `README.md` — Project documentation (this file)

## ✅ How to Run
1. Download the dataset from the Kaggle link above (`StudentsPerformance.csv`)
2. Open the script in Google Colab or Jupyter Notebook
3. Upload `StudentsPerformance.csv` to the same environment
4. Run all cells to reproduce the results

---
*Submitted as part of the Data Science Internship — SAM AI Technologies*
