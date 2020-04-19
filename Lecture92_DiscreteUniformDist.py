from __future__ import division
import numpy as np
from numpy.random import randn
import pandas as pd
from scipy import stats
from scipy.stats import randint
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


rollopts = [1,2,3,4,5,6]
tprob = 1
prob_roll = tprob/len(rollopts)


low,high = 1,7
mean, var = randint.stats(low,high)
plt.bar(rollopts,randint.pmf(rollopts,low,high))
plt.show()