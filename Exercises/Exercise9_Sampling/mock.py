import numpy as np 
from scipy.stats import norm, multivariate_normal

def lol():
    W = np.array([
        [1,1,2,2,3,3],
        [4,4,5,5,6,6],
        [7,7,8,8,9,9]
    ])

    lol = norm.pdf(W, np.array([1,2,3]).reshape(-1,1), np.array([3,4,5]).reshape(-1,1))
    print(lol)

    print(norm.pdf(W[0], 1, 3))
    print(norm.pdf(W[1], 2, 4))
    print(norm.pdf(W[2], 3, 5))

    print(np.zeros((2,1)))
    A = np.array([1,2,3,4])
    print(A.reshape(1, -1).repeat(50, axis=0))

def lol2():
    N = 100 
    means1 = np.random.rand(N, 3)*10
    means2 = np.random.rand(N, 3)*10
    
    print(means1.shape)
    print(means2.shape)
    sigma = np.eye(3)*10
    a1 = np.empty(N)
    for i, m1, m2 in zip(np.arange(N), means1, means2):
        a1[i]=multivariate_normal.pdf(m1, m2, sigma)
    
    a2 = np.empty(N)
    for i, m1, m2 in zip(np.arange(N), means1, means2):
        a2[i]=multivariate_normal.pdf(m2, m1, sigma)

    print(np.allclose(a1,a2))

lol2()