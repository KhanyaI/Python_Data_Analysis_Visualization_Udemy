import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn import metrics
from sklearn.svm import SVC
from sklearn import svm


iris = datasets.load_iris()
X = iris.data
Y = iris.target
print(X)

model = SVC()
X_train , X_test, Y_train, Y_test = train_test_split(X,Y)
model.fit(X_train,Y_train)
predicted = model.predict(X_test)
expected = Y_test
print (metrics.accuracy_score(expected,predicted))


X = iris.data[:,:2]
Y = iris.target
C = 1.0


svc = svm.SVC(kernel='linear', C=C).fit(X,Y)
rbf_svc = svm.SVC(kernel='rbf',C=C,gamma=0.7).fit(X,Y)
poly_svc = svm.SVC(kernel='poly',C=C,degree=3).fit(X,Y)
lin_svc = svm.LinearSVC(C=C).fit(X,Y)

h = 0.02
x_min = X[:,0].min()-1
x_max = X[:,0].min()+1

y_min = X[:,1].min()-1
y_max = X[:,1].max()+1

xx,yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
print(xx[:3],yy[:3])

"""
titles = ['SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel']
for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
	plt.figure(figsize=(15,15))
	plt.subplot(2,2,i+1)
	plt.subplots_adjust(wspace=0.4, hspace=0.4)
	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)

	plt.contourf(xx,yy,Z,cmap=plt.cm.terrain,alpha=0.5,linewidths =0)

	plt.scatter(X[:,0],X[:,1],c=Y,cmap = plt.cm.Dark2)
	plt.xlabel('Sepal length')
	plt.ylabel('Sepal width')
	plt.xlim(xx.min(), xx.max())
	plt.ylim(yy.min(), yy.max())
	plt.xticks(())
	plt.yticks(())
	plt.title(titles[i])

	plt.show()

"""
