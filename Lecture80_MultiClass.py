import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.datasets import load_iris
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
X = iris.data
print(X.shape)
Y = iris.target

iris_data = DataFrame(X,columns=['Sepal Length','Sepal Width','Petal Length','Petal Width'])
iris_target = DataFrame(Y,columns=['Species'])

def flower(num):
	if num ==0:
		return 'Setosa'
	elif num ==1:
		return 'Versicolor'
	else:
		return 'Virginica'

iris_target['Species'] = iris_target['Species'].apply(flower)

iris = pd.concat([iris_data, iris_target],axis=1)
print(iris.head())


#sns.pairplot(iris,hue='Species')


logreg = LogisticRegression()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4,random_state=3)
logreg.fit(X_train,Y_train)


Y_pred = logreg.predict(X_test)
metrics.accuracy_score(Y_test,Y_pred)



knn = KNeighborsClassifier(n_neighbors = 6)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
print(metrics.accuracy_score(Y_test,Y_pred))


krange = range(1,21)
accuracy = []

for k in krange:
	knn = KNeighborsClassifier(n_neighbors = k)
	knn.fit(X_train, Y_train)
	Y_pred = knn.predict(X_test)
	accuracy.append(metrics.accuracy_score(Y_test,Y_pred))

plt.plot(krange, accuracy)
plt.xlabel('K value for for kNN')
plt.ylabel('Testing Accuracy')
plt.show()
