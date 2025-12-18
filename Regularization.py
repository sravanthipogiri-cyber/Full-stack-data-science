# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 09:59:07 2025

@author: LENOVO
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression ,Ridge,Lasso
from sklearn.metrics import r2_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
dataset = pd.read_csv(r"D:\Spyder-12 th\car-mpg.csv")
dataset = dataset.drop(['car_name'],axis=1)
dataset['origin']=dataset['origin'].replace({1:'america',2:'europe',3:'asia'})
dataset = pd.get_dummies(dataset,columns=['origin'],dtype=int)
dataset= dataset.replace('?',np.nan)

dataset=dataset.apply(pd.to_numeric, errors='ignore')
numeric_cols=dataset.select_dtypes(include=[np.number]).columns
dataset[numeric_cols]=dataset[numeric_cols].apply(lambda x:x.fillna(x.median()))
dataset.head()
x=dataset.drop(['mpg'],axis=1)
y=dataset[['mpg']]
x_s = preprocessing.scale(x)
x_s = pd.DataFrame(x_s,columns=x.columns)
y_s = preprocessing.scale(y)
y_s = pd.DataFrame(y_s,columns=y.columns)
x_s
dataset.shape
x_train,x_test,y_train,y_test=train_test_split(x_s,y_s,test_size=0.20,random_state=0)
x_train.shape

regression_model= LinearRegression()
regression_model.fit(x_train,y_train)
for idx,col_name in enumerate (x_train.columns):
    print("The coefficient for {} is {}".format(col_name, regression_model.coef_[0][idx]))
    
intercept=regression_model.intercept_[0]
print('The intercept is {}'.format(intercept))
ridge_model=Ridge(alpha=0.4)
ridge_model.fit(x_train,y_train)
print('Ridge model coef:{}'.format(ridge_model.coef_))

lasso_model=Lasso(alpha=0.4)
lasso_model.fit(x_train,y_train)
print('Lasso model coef:{}'.format(lasso_model.coef_))



      











