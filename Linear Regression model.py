# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 09:39:57 2025

@author: LENOVO
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy.stats import variation
import scipy.stats as stats
import pickle




dataset=pd.read_csv(r"D:\Spyder-12 th\Salary_Data.csv")
x=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

regressor= LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
plt.scatter(x_test, y_test, color ='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Salary vs Experience(Test set)')
plt.xlabel('Years of Experiance')
plt.ylabel('Salary')
plt.show()
m_slope=regressor.coef_
print(m_slope)
c_intercept=regressor.intercept_
print(c_intercept)
y_20=m_slope*20+c_intercept
print(y_20)
y_12=m_slope*12+c_intercept
print(y_12)
bias_score=regressor.score(x_train, y_train)
print (bias_score)
variance_score=regressor.score(x_test,y_test)
print(variance_score)

#stats Integration to ML
dataset.mean()
dataset['Salary'].mean()
dataset.mode()
dataset.median()
dataset['YearsExperience'].mean()
dataset['Salary'].mean()

dataset.var()
dataset['YearsExperience'].var()
dataset.std()
dataset['Salary'].std()


variation (dataset.values)
variation(dataset['Salary'])
#corelation
dataset.corr()
dataset['Salary'].corr(dataset['YearsExperience'])

#Skewness
dataset.skew()
dataset['Salary'].skew()
#Standard Error
dataset.sem()
#z-Score

dataset.apply(stats.zscore)
stats.zscore(dataset['Salary'])
#Degree of freedom


#Sum of square regression or #ssr
y_mean=np.mean(y)
SSR=np.sum((y_pred-y_mean)**2)
print (SSR)

y=y[0:6]
SSE=np.sum((y-y_pred)**2)
print (SSE)

mean_total=np.mean(dataset.values)
SST = np.sum(( dataset.values-mean_total)**2)
print (SST)

filename='linear_regression_model.pkl'
with open (filename,'wb') as file:
    pickle.dump(regressor,file)
print("Model hasbeen Pickled and saved as linear_regression_model.pkl")









