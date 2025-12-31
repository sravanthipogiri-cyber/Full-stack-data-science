# -*- coding: utf-8 -*-
"""
Created on Wed Dec 31 10:18:30 2025

@author: LENOVO
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


# ===============================
# 2. LOAD TRAINING DATA
# ===============================
dataset = pd.read_csv(r"D:\2.LOGISTIC REGRESSION CODE\logit classification.csv")

# Encode categorical column (Gender)
dataset['Gender'] = dataset['Gender'].map({'Male': 1, 'Female': 0})

# Features & Target
X = dataset.iloc[:, [2, 3]].values   # example: Age, Gender
y = dataset.iloc[:, -1].values


# ===============================
# 3. TRAIN–TEST SPLIT
# ===============================
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=11
)


# ===============================
# 4. FEATURE SCALING (IMPORTANT)
# ===============================
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# ===============================
# 5. TRAIN LOGISTIC REGRESSION
# ===============================
classifier = LogisticRegression()
classifier.fit(x_train, y_train)


# ===============================
# 6. MODEL EVALUATION
# ===============================
y_pred = classifier.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

ac = accuracy_score(y_test, y_pred)
print("Accuracy:", ac)

cr = classification_report(y_test, y_pred)
print("Classification Report:\n", cr)

# (Note: these are TRAIN & TEST accuracy, not true bias/variance)
train_score = classifier.score(x_train, y_train)
test_score = classifier.score(x_test, y_test)

print("Training Accuracy:", train_score)
print("Testing Accuracy :", test_score)


# ===============================
# 7. LOAD NEW DATA FOR PREDICTION
# ===============================
dataset1 = pd.read_csv(r"D:\2.LOGISTIC REGRESSION CODE\final1.csv")

# Encode Gender SAME AS TRAINING
dataset1['Gender'] = dataset1['Gender'].map({'Male': 1, 'Female': 0})

# Keep full dataframe
d2 = dataset1.copy()

# Extract features
X_new = dataset1.iloc[:, [2, 3]].values

# Scale using SAME scaler (DO NOT FIT AGAIN)
X_new_scaled = sc.transform(X_new)


# ===============================
# 8. PREDICT ON NEW DATA
# ===============================
d2['y_pred1'] = classifier.predict(X_new_scaled)


# ===============================
# 9. SAVE RESULTS
# ===============================
d2.to_csv(r"D:\2.LOGISTIC REGRESSION CODE\final1_with_predictions.csv",
          index=False)

print("✅ Predictions saved successfully")

from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Predict probabilities (IMPORTANT)
y_pred_prob = classifier.predict_proba(x_test)[:, 1]

# AUC score
auc_score = roc_auc_score(y_test, y_pred_prob)
print("AUC Score:", auc_score)

# ROC curve values
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()
