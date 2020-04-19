import csv 
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from pandas_datareader import data, wb
from datetime import datetime

## General
tech_list =['AAPL','GOOGL','MSFT','AMZN']
end_date = datetime.now()
start_date=datetime(end_date.year-1,end_date.month,end_date.day)

for stock in tech_list:
	globals()[stock] = data.DataReader(stock,data_source='yahoo',start=start_date)



## What was the change in price of the stock over time?

#AAPL['Adj Close'].plot()


## What was the daily return of the stock on average?

AAPL['Daily return'] = AAPL['Adj Close'].pct_change()
#AAPL['Daily return'].plot(figsize=(10,4),legend=True,linestyle='dashed', marker='o')

## What was the moving average of the various stocks?
movingavg_lst = [10,20,50]

for ma in movingavg_lst:
	column_name = 'MA for %s days'%(str(ma))
	AAPL[column_name] = AAPL['Adj Close'].rolling(ma).mean()


#AAPL[['Adj Close','MA for 10 days', 'MA for 20 days','MA for 50 days']].plot(figsize=(10,4))

## What was the correlation between different stocks' closing prices?

closingprice_df = data.DataReader(tech_list,data_source='yahoo',start=start_date)['Adj Close']
tech_rets = closingprice_df.pct_change()


## What was the correlation between different stocks' daily returns?

"""
sns.jointplot(AAPL, MSFT,tech_rets,kind='scatter',height=5,color='g')
sns.pairplot(tech_rets.dropna())
returnfig = sns.PairGrid(tech_rets.dropna())
returnfig.map_upper(plt.scatter,color='purple')
returnfig.map_lower(sns.kdeplot,color='r')
returnfig.map_diag(plt.hist,bins=30)



sns.heatmap(closingprice_df.dropna().corr())
"""

## How much value do we put at risk by investing in a particular stock?

rets = tech_rets.dropna()

"""
area = np.pi*20
plt.scatter(rets.mean(),rets.std(),s=area)
plt.xlabel('Expected Return')
plt.ylabel('Risk')

for label, x, y in zip(rets.columns,rets.mean(),rets.std()):
	plt.annotate(label,xy=(x,y))
"""

## How can we attempt to predict future stock behavior?
#AAPL['Daily return'].quantile(0.05)

print(GOOGL['Open'].head(5))

days = 365
dt = 1/days
mu = rets.mean()['GOOGL']
sigma = rets.std()['GOOGL']


def stock_montecarlo(start_price, days, mu, sigma,dt):
	price = np.zeros(days)
	price[0] = start_price
	shock = np.zeros(days)
	drift = np.zeros(days)


	for x in range(1,days):
		shock[x] = np.random.normal(loc=mu*dt, scale=sigma*np.sqrt(dt)) 
		drift[x] = mu*dt
		price[x] = price[x-1]+(price[x-1]*(drift[x]+shock[x])) 

	return price

start_price = 1187.540039

#for x in range(100):
	#plt.plot(stock_montecarlo(start_price,days,mu,sigma,dt))

runs = 1000

sims = np.zeros(runs)

for i in range(runs):
	sims[i] = stock_montecarlo(start_price,days,mu,sigma,dt)[days-1]

q = np.percentile(sims,1)
#plt.hist(sims,bins=100)

###################################

### Additional
##Estimate the values at risk using both methods we learned in this project for a stock not related to technology.
ticker =['JNJ']
end_date = datetime.now()
start_date=datetime(end_date.year-1,end_date.month,end_date.day)

stock = data.DataReader(stock,data_source='yahoo',start=start_date)

returns = stock['Adj Close'].pct_change().dropna()
#plt.scatter(returns.mean(),returns.std())
#plt.xlabel('Expected Return')
#plt.ylabel('Risk')

#2nd way: 
print(returns.quantile(0.05))


