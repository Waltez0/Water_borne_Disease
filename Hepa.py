# -*- coding: utf-8 -*-
"""Hepatitis.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1U4JnDy_U0qfLUA_BPk0-A7T0Qh-OO9gi
"""

import joblib
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LassoCV
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import GridSearchCV
import pickle

data_df=pd.read_csv('hepatitis.csv',header=0)
data_df

data_df.dtypes

X = data_df.drop("liver_big",1)
y = data_df["liver_big"]
regressor=LassoCV()
regressor.fit(X,y)

#print(regressor.coef_)
coef_col = pd.Series(regressor.coef_,index=X.columns)
#print(coef_col)
sorted_coef = coef_col.sort_values()
matplotlib.rcParams['figure.figsize']=(10.0,7.0)
sorted_coef.plot(kind="barh")
no_of_selected_features = sum(coef_col != 0)
no_of_rejected_features = sum(coef_col == 0)
total_features = no_of_selected_features + no_of_rejected_features
print("L1 selected only ",no_of_selected_features," features out of ",total_features," from the transformed dataset ")
plt.title("Feature Selection Using L1 Embedder")

data_df.columns

feature_df = data_df[['age',  'steroid', 'liver_firm',  'spiders'
       ]]

def svm_pred(age , steroid, spiders, liver_firm):
 data = feature_df
 
 x = np.asarray(feature_df)
 y = np.asarray(data['liver_big'])

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4 )

classifier = svm.SVC(kernel='linear', gamma='auto', C=5)
classifier.fit(X_train, y_train)
y_predict = classifier.predict(X_test)

print(classification_report(y_test, y_predict))

plot_confusion_matrix(classifier,
                 X_test,
                 y_test,
                 values_format = 'd',
                display_labels=["liver big", "small"])

