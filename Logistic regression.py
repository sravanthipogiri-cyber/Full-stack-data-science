# -*- coding: utf-8 -*-
"""
Created on Tue Dec 30 10:04:19 2025

@author: LENOVO
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


dataset=pd.read_csv(r"D:\logit classification.csv")
x = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:, -1].values


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=11)


#sc=StandardScaler()
#x_train=sc.fit_transform(x_train)
#x_test=sc.transform(x_test)


classifier=LogisticRegression()
classifier.fit(x_train, y_train)

y_pred=classifier.predict(x_test)

cm=confusion_matrix(y_test, y_pred)
print(cm)

ac=accuracy_score(y_test,y_pred)
print(ac)


cr = classification_report(y_test, y_pred)
cr

bias = classifier.score(x_train, y_train)
bias

variance = classifier.score(x_test, y_test)
variance







