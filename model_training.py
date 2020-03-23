#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 02:24:46 2019

@author: cursedomonstro
"""
import numpy as np
import matplotlib.pyplot as plt

X = 2*np.random.rand(100,1)
y=4+3*X + np.random.randn(100,1)

X_b = np.c_[np.ones(len(X)),X]

theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y) # loi normal test

y_predict = X_b.dot(theta)

plt.plot(X, y, 'r.', label = 'true lab')
plt.plot(X, y_predict, label='prediction')
plt.legend(loc="upper left")