# =========================================================
# TASK 1: Heart Disease Prediction
# =========================================================
# এই কোডটা ধাপে ধাপে কাজ করে:
# 1. ডেটা লোড করা
# 2. ডেটা বোঝা ও পরিষ্কার করা (preprocessing)
# 3. একাধিক মডেল ট্রেইন করা ও তুলনা করা
# 4. মডেল evaluate করা (Accuracy, Precision, Recall, F1)
# =========================================================

# ---- Step 1: প্রয়োজনীয় লাইব্রেরি ইম্পোর্ট ----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ---- Step 2: ডেটা লোড করা ----
# NOTE: 'heart.csv' ফাইলটা Colab-এ আপলোড করা থাকতে হবে (বাম পাশের ফোল্ডার আইকন দিয়ে)
df = pd.read_csv('heart.csv')

print("ডেটার প্রথম ৫টা রো দেখি:")
print(df.head())

print("\nডেটার shape (কতগুলো রো, কতগুলো কলাম):", df.shape)

print("\nকলামগুলোর তথ্য:")
print(df.info())

print("\nকোনো missing value আছে কিনা:")
print(df.isnull().sum())

# ---- Step 3: Target column চেক করা ----
# এই dataset-এ সাধারণত 'target' কলামটা বলে দেয় patient-এর heart disease আছে (1) নাকি নেই (0)
print("\nTarget column-এর value counts:")
print(df['target'].value_counts())

# ---- Step 4: Features (X) এবং Target (y) আলাদা করা ----
X = df.drop('target', axis=1)   # target বাদে বাকি সব কলাম = features
y = df['target']                 # শুধু target column

# ---- Step 5: Train/Test split করা ----
# ডেটার ৮০% দিয়ে মডেল শেখাবো (train), ২০% দিয়ে টেস্ট করবো
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining data: {X_train.shape[0]} রো")
print(f"Testing data: {X_test.shape[0]} রো")

# ---- Step 6: Feature Scaling ----
# কিছু মডেল (যেমন Logistic Regression) সংখ্যাগুলো একই স্কেলে থাকলে ভালো কাজ করে
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---- Step 7: একাধিক মডেল ট্রেইন করা ও তুলনা করা ----
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

results = {}

for name, model in models.items():
    # Logistic Regression scaled data দিয়ে, বাকিগুলো normal data দিয়ে ভালো কাজ করে
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results[name] = {
        "Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1
    }

    print(f"\n===== {name} =====")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1-Score : {f1:.3f}")

# ---- Step 8: সব মডেলের ফলাফল একটা টেবিলে দেখা ----
results_df = pd.DataFrame(results).T
print("\n===== সব মডেলের তুলনা =====")
print(results_df)

# ---- Step 9: সবচেয়ে ভালো মডেলের Confusion Matrix দেখা ----
best_model_name = results_df['F1-Score'].idxmax()
print(f"\nসবচেয়ে ভালো মডেল (F1-Score অনুযায়ী): {best_model_name}")

best_model = models[best_model_name]
if best_model_name == "Logistic Regression":
    y_pred_best = best_model.predict(X_test_scaled)
else:
    y_pred_best = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease'])
plt.title(f'Confusion Matrix - {best_model_name}')
plt.ylabel('আসল মান (Actual)')
plt.xlabel('মডেলের প্রেডিকশন (Predicted)')
plt.show()

# ---- Step 10: Feature Importance দেখা (Random Forest দিয়ে) ----
rf_model = models["Random Forest"]
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(x=importances.values, y=importances.index)
plt.title("কোন Feature গুলো heart disease predict করতে বেশি গুরুত্বপূর্ণ")
plt.xlabel("Importance Score")
plt.show()

print("\n===== সম্পূর্ণ Classification Report (Best Model) =====")
print(classification_report(y_test, y_pred_best))
