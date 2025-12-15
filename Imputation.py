# -*- coding: utf-8 -*-
"""
Created on Sun Dec 14 23:09:12 2025

@author: LENOVO
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset=pd.read_csv(r"D:\Spyder-12 th\data.csv")
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values
from sklearn.impute import SimpleImputer
imputer=SimpleImputer()
imputer-imputer.fit(x[:,1:3])
x[:,1:3]=imputer.transform(x[:,1:3])
    