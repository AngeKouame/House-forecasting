#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 00:46:19 2019

@author: cursedomonstro
"""
from sklearn import datasets
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import precision_recall_curve, confusion_matrix, precision_score, recall_score,f1_score

mnist = datasets.load_digits()
X, y = mnist['data'], mnist['target']


X_train, X_test, y_train, y_test = X[:1200], X[1200:], y[:1200], y[1200:]

shuffle_index = np.random.permutation(1200)
X_train = X_train[shuffle_index]
y_train = y_train[shuffle_index]

y_train_5 = (y_train==5)
y_test_5 = (y_test==5)


model = SGDClassifier(random_state=42)
model.fit(X_train, y_train_5)
y_scores = cross_val_predict(model, X_train, y_train_5, cv=5, method="decision_function")

scores = confusion_matrix(y_train_5, model.predict(X_train))
print(precision_score(y_train_5, model.predict(X_train)))
print(recall_score(y_train_5, model.predict(X_train)))
print(f1_score(y_train_5, model.predict(X_train)))
precision, recall, treshold = precision_recall_curve(y_train_5, y_scores)

plt.plot(precision[:-1], recall[:-1],"b",label="precision")

plt.legend(loc="upper right")