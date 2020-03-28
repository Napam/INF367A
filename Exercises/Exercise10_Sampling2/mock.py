import numpy as np 

A = np.array([
    [1,2,np.nan],
    [np.nan,2,1],
    [1,2,np.nan],
    [np.nan,np.nan,1],
])

mask = np.isnan(A)
A[mask] = np.arange(mask.sum())
print(A)