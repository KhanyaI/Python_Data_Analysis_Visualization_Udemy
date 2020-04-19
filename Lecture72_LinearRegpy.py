
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression

TK_SILENCE_DEPRECATION=1

boston = load_boston()
#print (boston.DESCR)

"""
plt.hist(boston.target,bins=50)
plt.xlabel('Prices')
plt.ylabel('No of houses')
"""

#plt.scatter(boston.data[:,5],boston.target)

boston_df = DataFrame(boston.data)
boston_df.columns = boston.feature_names
boston_df['Price'] = boston.target


#sns.lmplot('RM','Price',data=boston_df)


print(boston_df.RM.head())
X = np.vstack(boston_df.RM)
X = np.array([[value,1] for value in X], dtype=float) ## 1st step to make it into a matrix
Y = boston_df.Price

m,b = np.linalg.lstsq(X,Y)[0] ## 2nd step to make it into a matrix
#plt.plot(boston_df.RM,boston_df.Price,'o')
x = boston_df.RM
#plt.plot(x,m*x+b,'r',label='Best Fit Line')



results = np.linalg.lstsq(X,Y,rcond=None)
error_total = results[1]
rmse = np.sqrt(error_total/len(X))

### Multivariable LR

lreg = LinearRegression()
Xmulti = boston_df.drop('Price',1)
Ytarget = boston_df.Price

lreg.fit(Xmulti,Ytarget)
lreg.intercept_,len(lreg.coef_)

coeffdf = DataFrame(boston_df.columns)
coeffdf['Coeff Estimates'] = Series(lreg.coef_)

## Testing and Validation
X_test, X_train, Y_test, Y_train = train_test_split(X, boston_df.Price)
lreg.fit(X_train,Y_train)
pred_Train = lreg.predict(X_train)
pred_Test = lreg.predict(X_test)
mean_sq_error = np.mean((Y_train - pred_Train)**2)
print(mean_sq_error)

mean_sq_error_test = np.mean((Y_test - pred_Test)**2)
print(mean_sq_error_test)

train = plt.scatter(pred_Train,(pred_Train - Y_train), c = 'b', alpha=0.5)

test = plt.scatter(pred_Test,(pred_Test - Y_test), c = 'r', alpha=0.5)
plt.show()

