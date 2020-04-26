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
A = abs(A / A.max()) * 6
A = A.astype(int)
# A = sparse.random(10, 6, format='dense', density=0.5, dtype=int) % 6
B = np.array([[0,0,4,2,2,3]])
P = PCA(4)
P.fit(A)
X = (A-A.mean(axis=0))@P.components_.T
R = (X@P.components_+P.mean_)

print('B:', B.shape)
print(B)
print()

print('A:',A.shape)
print(A)
print()

print('P:', P.components_.shape)
print(P.components_)
print()

print('X:', X.shape)
print(X)
print()

print('R:', R.shape)
print(R)
print()

pred = (B-P.mean_)@P.components_.T
pred = pred@P.components_+P.mean_

print('Pred: ', pred.shape)
print(pred)




