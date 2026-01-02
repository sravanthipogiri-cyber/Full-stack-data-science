# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 10:01:28 2026

@author: LENOVO
"""

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

from sklearn.svm import SVC
classifier=SVC()
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# This is to get the Models Accuracy
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)

bias = classifier.score(x_train, y_train)
print(bias)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# This is to get the Models Accuracy
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)

bias = classifier.score(x_train, y_train)
print(bias)

variance = classifier.score(x_test, y_test)
print(variance)
from sklearn.svm import SVC
classifier = SVC(C = 1.0, kernel='poly', degree=3)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# This is to get the Models Accuracy
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)

bias = classifier.score(x_train, y_train)
print(bias)


