import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 
from scipy.stats import multivariate_normal, norm

def get_grid(n=100, xrange=(-10,10), yrange=(-10,10)):
    xspace = np.linspace(*xrange, n)
    yspace = np.linspace(*yrange, n)
    Xmesh, Ymesh = np.meshgrid(xspace, yspace)
    points = np.array([Xmesh.ravel(), Ymesh.ravel()]).T
    return points 

# def get_densitymap(func, funcargs=(), funckwargs={}, resolution=224, xrange=(-10,10), yrange=(-10,10)):
#     points = get_grid(resolution, xrange, yrange)
#     z = func(points, *funcargs, **funckwargs)
#     return z.reshape((resolution, resolution)), points

if __name__ == '__main__':
    df = pd.read_csv('new_york_bicycles2.csv')
    # print(df)
    X = np.column_stack([df.values[:,0], np.ones(len(df))])
    y = df.values[:,1]
    y_col = y.reshape(-1,1)

    beta = 1
    alpha_= 1

    S = np.linalg.inv(alpha_*np.eye(2) + beta * X.T@X)
    m = beta*S @ (X*y_col).sum(axis=0)

    # print(m, S)

    n=100
    points = get_grid(n, xrange=(1.82,1.84), yrange=(429.9,430.1))

    z = multivariate_normal.pdf(points, m, S)
    # best_point = np.argmax(z)
    # print(points[best_point])

    # plt.imshow(z.reshape((n,n)), extent=[1.82,1.84,430.1,430.2], origin='lower')
    # plt.show()

    linfunc = lambda x: x*m[0] + m[1]

    plt.scatter(*df.values.T)
    xrange = np.linspace(0,5000,100)
    pred = linfunc(xrange)
    plt.plot(xrange, pred, color='r')

    lol = [] 
    for x, y in zip(xrange, pred):
        x_ = np.array([x,y]).reshape(-1,1)
        lol.append(norm.ppf(x, m@x_, beta + (x_.T@S@x_).ravel()[0]))

    print(lol)
    exit()
    plt.plot(xrange, lol, color='r')
    plt.show()





