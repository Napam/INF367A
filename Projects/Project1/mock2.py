from scipy.stats import gamma, norm
import numpy as np 
from matplotlib import pyplot as plt 
from sklearn.preprocessing import PolynomialFeatures

A = np.array([['a','b','c']])
B = np.array([['z','x','a','e','b','d','g','c','p']])
C = np.array([1,2,3,4,5,6,7,8,9])

G = np.repeat(B, 3, axis=0)
print((G == np.repeat(A.T, B.shape[1], axis=1)))

# for i in range(A):
    # for j in range(B):
        # if A[i] == B[j]:
            # A 


