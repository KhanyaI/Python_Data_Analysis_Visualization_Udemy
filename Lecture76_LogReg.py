import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import statsmodels.api as sm
import math
TK_SILENCE_DEPRECATION=1



df = sm.datasets.fair.load_pandas().data

def affair_check(x):
    if x != 0:
        return 1
    else:
        return 0

df['Had_Affair'] = df['affairs'].apply(affair_check)

df.groupby(['Had_Affair']).mean()

#sns.factorplot('yrs_married',data=df,hue='Had_Affair',kind='count') #do for edu, no_child and age

occ_dummies = pd.get_dummies(df['occupation'])
hus_occ_dummies = pd.get_dummies(df['occupation_husb'])
occ_dummies.columns = ['occ1','occ2','occ3','occ4','occ5','occ6']
hus_occ_dummies.columns = ['hocc1','hocc2','hocc3','hocc4','hocc5','hocc6']


X = df.drop(['occupation','occupation_husb','Had_Affair'],axis=1)
Y = df.Had_Affair
occ_dummies = pd.concat([occ_dummies,hus_occ_dummies],axis=1)
X = pd.concat([X,occ_dummies],axis=1)


##Multicollinarilty 
X = X.drop('occ1',axis=1)
X = X.drop('hocc1',axis=1)
X = X.drop('affairs',axis=1)
Y = np.ravel(Y) #flatten to 1D
print(type(X), type(Y))

## Sklearn
log_model = LogisticRegression()
log_model.fit(X,Y)
log_model.score(X,Y)

coeff_df =  DataFrame(zip(X.columns,np.transpose(log_model.coef_)))

X_train, X_test, Y_train, Y_test = train_test_split(X,Y)
log_model2 = LogisticRegression()
log_model2.fit(X_train,Y_train)
class_predict = log_model2.predict(X_test)
print(metrics.accuracy_score(Y_test,class_predict))



