import pandas as pd
from pandas import Series,DataFrame

import matplotlib.pyplot as plt
import seaborn as sns


from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split


iris = datasets.load_iris()
X = iris.data
Y = iris.target
model = GaussianNB()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
model.fit(X_train,Y_train)
predicted = model.predict(X_test)
expected = Y_test