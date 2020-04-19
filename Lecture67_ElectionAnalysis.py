
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import requests
from io import StringIO
from datetime import datetime
##General
url = "https://elections.huffingtonpost.com/pollster/2012-general-election-romney-vs-obama.csv"
source = requests.get(url).text
polldata = StringIO(source)

poll_df = pd.read_csv(polldata)
#print(poll_df.info())

##Who was being polled and what was their party affiliation?
#sns.factorplot('Affiliation',data = poll_df,kind='count',hue='Population')
avg = pd.DataFrame(poll_df.mean())
avg.drop('Number of Observations',axis=0,inplace=True)
std = pd.DataFrame(poll_df.std())
std.drop('Number of Observations',axis=0,inplace=True)

#avg.plot(yerr=std,kind='bar',legend=False)
pollavg = pd.concat([avg,std],axis=1)
pollavg.columns = ['Avg','Std']



## Did the poll results favor Romney or Obama?
#poll_df.plot(x='End Date',y=['Obama','Romney','Undecided'])

poll_df['Difference'] = (poll_df['Obama'] - poll_df['Romney'])/100
poll_df = poll_df.groupby(['Start Date'],as_index=False).mean()
#poll_df.plot('Start Date','Difference',figsize=(10,4),marker='o')
rowin = 0
xlimt = []

##Can we see an effect in the polls from the debates?
for date in poll_df['Start Date']:
	if date[0:7] == '2012-10':
		xlimt.append(rowin)
		rowin +=1
	else:
		rowin +=1
minx = min(xlimt)
maxx = max(xlimt)

#poll_df.plot('Start Date','Difference',figsize=(10,4),marker='o',xlim=(329,356))


donordf = pd.read_csv('Election_Donor_Data.csv')
amt_money_contributed = donordf['contb_receipt_amt'].value_counts()
donor_mean = donordf['contb_receipt_amt'].mean()
donor_std = donordf['contb_receipt_amt'].std()
top_donor = donordf['contb_receipt_amt'].copy()
top_donor.sort_values(inplace=True)

top_donor = top_donor[top_donor > 0]
top_donor.sort_values(inplace=True)


candidates = donordf.cand_nm.unique()
donordf = donordf[donordf['contb_receipt_amt']>0]
donordf.groupby('cand_nm')['contb_receipt_amt'].count()
cand_amt = donordf.groupby('cand_nm')['contb_receipt_amt'].sum()

party_map = {'Bachmann, Michelle': 'Republican',
           'Cain, Herman': 'Republican',
           'Gingrich, Newt': 'Republican',
           'Huntsman, Jon': 'Republican',
           'Johnson, Gary Earl': 'Republican',
           'McCotter, Thaddeus G': 'Republican',
           'Obama, Barack': 'Democrat',
           'Paul, Ron': 'Republican',
           'Pawlenty, Timothy': 'Republican',
           'Perry, Rick': 'Republican',
           "Roemer, Charles E. 'Buddy' III": 'Republican',
           'Romney, Mitt': 'Republican',
           'Santorum, Rick': 'Republican'}


donordf['Party'] = donordf.cand_nm.map(party_map)
cand_amt.plot(kind='bar')

occupationdf = donordf.pivot_table('contb_receipt_amt',index='contbr_occupation',columns='Party',aggfunc = 'sum')
occupationdf.loc['CEO'] = occupationdf.loc['CEO']+occupationdf.loc['C.E.O.']
occupationdf.drop(['C.E.O.','INFORMATION REQUESTED PER BEST EFFORTS','INFORMATION REQUESTED'],inplace=True)
rich_occupation = occupationdf[occupationdf.sum(1) > 1000000]
rich_occupation.plot(kind='barh',figsize=(12,14))
plt.show()
##How do undecided voters effect the poll?
##Can we account for the undecided voters?
##How did voter sentiment change over time?
