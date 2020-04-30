import numpy as np
seed=42069
np.random.seed(seed)
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
import sys 
from scipy import sparse 
from matplotlib import pyplot as plt 


np.set_printoptions(precision=2, suppress=True)

A, y = make_blobs(n_samples=12, n_features=6)

import re 

txt = '''
import numpy as              np
// fuck lol fasdfasd fasdf asdf asdf asdf
// adfads fadsf asdf asdffasdf 

pd.DataFrame

// asdfas dfadfa 
// adfasdf adsf asdf asdf as

np.arange(10)
'''
txt = re.sub('(//.+)', '', txt)
txt = re.sub('(\n{2,})', '\n', txt)
txt = re.sub('([ \t]{2,})', ' ', txt)

print(txt)



