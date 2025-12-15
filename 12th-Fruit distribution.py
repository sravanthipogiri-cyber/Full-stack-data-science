# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 10:01:37 2025

@author: LENOVO
"""

import matplotlib.pyplot as plt

sizes=[40,25,20,15]
labels=['Python','java','golang','c']
plt.pie (sizes,labels=labels,autopct='%1.1f%%', startangle=90)
plt.title("Fruit Distribuyion")
plt.show()