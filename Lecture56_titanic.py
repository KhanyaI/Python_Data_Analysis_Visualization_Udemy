import csv 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.DataFrame(pd.read_csv('/Users/ifrahkhanyaree/Desktop/HomeDS/Code/Kaggle/titanic/train.csv'))
#print(df.info())
#print(df.columns)


## Who were the passengers? 

#sns.factorplot('Sex', data = df,kind='count')#allows to pass column argument
#sns.factorplot('Pclass', data = df,kind='count',hue='Sex')#allows to pass column argument

#plt.show()

def male_female_child(passenger):
	age,sex = passenger

	if age < 16:
		return 'child'
	else:
		return sex

df['person'] = df[['Age','Sex']].apply(male_female_child,axis=1)
#print(df.person.head(5))

"""
df['Age'].hist(bins=100)
fig = sns.FacetGrid(df, hue='person',aspect = 4)
fig.map(sns.kdeplot,'Age',shade=True)
oldest = df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()


sns.factorplot('Embarked',data=df,hue='Pclass',kind='count')
plt.show()
"""

## Who was alone?

df['Alone'] = df.SibSp + df.Parch

for i in range(len(df.Alone)):
	if df.Alone[i] == 0:
		df.Alone[i] = 'Alone'
	else:
		df.Alone[i] = 'With family'

#sns.factorplot('Alone', data=df,kind='count',palette='Blues')


## Survivors
## Did having a family member increase your chances of survival? 

df['Survivors'] = df.Survived.map({0:'no',1:'yes'})
#sns.factorplot('Survivors', data=df,kind='count',palette='Blues')


#sns.lmplot('Age','Survived',data=df,palette='winter')


## Did cabin have an effect on passenger survival rate?

decklist=[]
df.dropna(subset=['Cabin'],inplace=True)

for letter in df.Cabin:
	decklist.append(letter[0])

df['Deck'] = decklist
#print(df.Deck.head(5))
#sns.factorplot('Deck',data=df,hue='Survived',kind='count')



