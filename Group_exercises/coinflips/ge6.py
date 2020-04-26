import numpy as np 
import pandas as pd 
import seaborn as sns 
from matplotlib import pyplot as plt 
from scipy.stats import beta, bernoulli
import re
from itertools import groupby

df = pd.read_csv('coin_flips.csv', header=None)
X = df.values.flatten()

coin_types, counts = np.unique(X, return_counts=True)
print(f'Coin types {coin_types}')
print(f'Coin counts {counts}')

a = len(X)//2
b = len(X)//2
xrange = np.linspace(0, 1, 1000)
y = beta.pdf(xrange, a+X.sum(), b + len(X) - X.sum())
theta_ = xrange[y.argmax()]
X_ = bernoulli.rvs(theta_, size=len(X))

print(re.findall('(1+)',str(X_).replace(' ','')))
print(re.findall('(0+)',str(X_).replace(' ','')))


# fig, (ax_left, ax_right) = plt.subplots(1,2, sharey=True)
# sns.countplot(X, ax=ax_left)
# sns.countplot(X_, ax=ax_right)
# plt.show()

# p = np.clip(p, epsilon, 1)
# return p @ np.log2(1/p)'
