#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 16:06:18 2018

@author: anirban
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, 2].values

#Creating the linear regressor model
from sklearn.linear_model import LinearRegression
lin_reg1 = LinearRegression()
lin_reg1.fit(X,y)

#Creating the polynomial regressor
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)

#Plotting the Linear Regressor
plt.scatter(X, y, color = 'red')
plt.plot(X,lin_reg1.predict(X),color = 'blue')
plt.title('Truth vs Bluff (Linear Regression)')
plt.xlabel('Position label')
plt.ylabel('Salary')
plt.show()

#Plotting the Polynomial Regressor
plt.scatter(X, y, color = 'yellow')
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.plot(X_grid,lin_reg2.predict((poly_reg.fit_transform(X_grid))),color = 'black')
plt.title('Truth vs Bluff (Polynomial Regression)')
plt.xlabel('Position label')
plt.ylabel('Salary')
plt.show()

#predicting Values
lin_reg1.predict(6.5)
lin_reg2.predict((poly_reg.fit_transform(6.5)))
