import numpy as np 
seed = 42069
np.random.seed(seed)
import pandas as pd 
from matplotlib import pyplot as plt 
from scipy.stats import multivariate_normal

def p(theta: np.ndarray, X: np.ndarray, y: np.ndarray, alpha: float=1):
    term1 = -0.5*np.linalg.multi_dot([theta.T, alpha*np.eye(len(theta)), theta])
    term2 = theta.T @ (X*y).sum(axis=0)
    term3 = -sum([np.e ** (theta.T @ x) for x in X])
    return term1 + term2 + term3

def dp(theta: np.ndarray, X: np.ndarray, y: np.ndarray, alpha: float=1):
    term1 = -(alpha * np.eye(len(theta))) @ theta 
    term2 = (X*y).sum(axis=0).reshape(-1,1) 
    term3 = -sum([x * np.e ** (theta.T @ x) for x in X]).reshape(-1,1)
    return term1 + term2 + term3

def ddp(theta: np.ndarray, X: np.ndarray, y: np.ndarray, alpha: float=1):
    term1 = -alpha * np.eye(len(theta))
    X = X[:,np.newaxis,:]
    term2 = -sum([(x.T@x) * np.e ** (theta.T @ x.T) for x in X]) 
    return term1 + term2

def laplace_iteration(theta: np.ndarray, X: np.ndarray, y: np.ndarray):
    dE = dp(theta, X, y)
    H_inv = np.linalg.inv(ddp(theta, X, y))
    return theta - H_inv@dE, dE, H_inv

def get_grid(n=100, xrange=(-10,10), yrange=(-10,10)):
    xspace = np.linspace(*xrange, n)
    yspace = np.linspace(*yrange, n)
    Xmesh, Ymesh = np.meshgrid(xspace, yspace)
    points = np.array([Xmesh.ravel(), Ymesh.ravel()]).T
    return points 

def get_densitymap(func, funcargs=(), funckwargs={}, resolution=224, xrange=(-10,10), yrange=(-10,10)):
    points = get_grid(resolution, xrange, yrange)
    z = func(points, *funcargs, **funckwargs)
    return z.reshape((resolution, resolution)), points

if __name__ == '__main__':
    df = pd.read_csv('new_york_bicycles.csv', header=None).sample(frac=1, random_state=seed)
    X = np.column_stack([df.values[:,0], np.ones(len(df))])
    y = df.values[:,1]
    
    N_train = 150
    
    train_slice = slice(0, N_train)
    test_slice = slice(N_train, None)

    X_train = X[train_slice]
    X_test = X[test_slice]
    
    y_train = y[train_slice]
    y_test = y[test_slice]
    y_col = y.reshape(-1,1)

    def task2d(show=True):
        theta = np.array([[0,0]]).T

        for i in range(100):
            theta_new, dE, H_inv = laplace_iteration(theta, X, y_col)
            if np.allclose(theta_new, theta, rtol=1e-10, atol=1e-10): 
                print(f'Converged at iteration {i}: {theta_new.ravel()}')
                break
            theta = theta_new

        return theta.ravel(), H_inv

    def task2e(show=True):
        mu, Sigma = task2d(False)

        lol = dp(mu.reshape(-1,1), X, y_col)
        print(np.linalg.inv(ddp(lol.reshape(-1,1), X, y_col)))
        print(np.linalg.inv(ddp(mu.reshape(-1,1), X, y_col)))
        # print(ddp(lol.reshape(-1,1), X, y_col))
        # print(ddp(mu.reshape(-1,1), X, y_col))

        # Try the thing
        # multivariate_normal.pdf(np.array([[0,0]]), mean=mu, cov=Sigma)

        points = get_grid()

        if show:
            points = get_grid()

            z_pri = multivariate_normal.pdf(points, m0, s0)
            z_post = multivariate_normal.pdf(points, mu_n, sigma_n)            

            fig, (ax1, ax2) = plt.subplots(1,2, figsize=(7,3))
            fig.suptitle('Actual vs Laplace approximation')
            ax1.set_title('Actual')
            ax1.imshow(z_pri.reshape((resolution,resolution)), extent=[8, 20, 12, 22], origin='lower')

            ax2.set_title('Laplace')
            ax2.imshow(z_post.reshape((resolution, resolution)), extent=[8, 20, 12, 22], origin='lower')
            
            plt.setp([ax1, ax2], xlabel='x', ylabel='y')
            
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])

            maxima2d(ax2, points, z_post)
            maxima2d(ax1, points, z_pri)

            # plt.savefig('images/2c.pdf')
            plt.show()
            pass

    # task2d(False)
    task2e(False)
    
