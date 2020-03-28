import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 
from scipy.stats import multivariate_normal, gamma, norm, rv_histogram
np.random.seed(42069)

class LinRegGibbs:
    def __init__(self, alpha=1, beta=1, a=1, b=1):
        self.alpha = alpha 
        self.beta = beta 
        self.a = a
        self.b = b

    def fit(self, X, y, n=69000):
        self.n = n
        self.X = X 
        self.y = y
        self.ws = np.empty((n, 2))
        self.betas = np.empty(n)

        I = np.eye(2)
        self.ws[0] = np.array([1,1])
        self.betas[0] = self.beta
        origo = np.zeros(2)
        for i in range(1, n):
            S = np.linalg.inv(I*self.alpha + self.betas[i-1]*X.T@X)
            m = self.betas[i-1]*S@(X*y.reshape(-1,1)).sum(0)
            self.ws[i] = multivariate_normal.rvs(mean=m, cov=S, size=1)

            p = y - self.ws[i]@X.T
            self.betas[i] = gamma.rvs(self.a+len(X)/2,1/(self.b+0.5*p.T@p))
    
    def posterior(self):
        epsilon = 1e-12
        I = np.eye(2)
        origo = np.zeros(2)
        logw = np.log(multivariate_normal.pdf(self.ws, origo, self.alpha*I).clip(epsilon))
        logbeta = np.log(gamma.pdf(self.betas, self.a, self.b).clip(epsilon))
        logdata =\
            np.log(norm.pdf(
                x=self.y-self.ws@X.T, 
                loc=np.zeros((self.n,1)), 
                scale=self.betas.reshape(-1,1)
            ).clip(epsilon)).sum()
        self.logposterior = logw+logbeta+logdata

    def plot_result(self):
        self.posterior()
        fig, axes = plt.subplots(4,2, figsize=(7,6))
        ax_left = axes[:,0].ravel()
        ax_right = axes[:,1].ravel()
        ax_left[0].plot(self.logposterior, linewidth=0.2)
        ax_left[0].set_title(r'Posterior trace')
        ax_left[1].plot(self.ws[:,0], linewidth=0.2)
        ax_left[1].set_title(r'$w_1^t$ (slope) trace')
        ax_left[2].plot(self.ws[:,1], linewidth=0.2)
        ax_left[2].set_title(r'$w_0^t$ (intercept) trace')
        ax_left[3].plot(self.betas, linewidth=0.2)
        ax_left[3].set_title(r'$\beta^t trace$')
        
        ax_right[0].hist(self.logposterior, bins=40)
        ax_right[0].set_title(r'Posterior hist')
        ax_right[1].hist(self.ws[:,0], bins=40)
        ax_right[1].set_title(r'$w_1^t$ hist')
        ax_right[2].hist(self.ws[:,1], bins=40)
        ax_right[2].set_title(r'$w_0^t$ hist')
        ax_right[3].hist(self.betas, bins=40)
        ax_right[3].set_title(r'$\beta^t$ hist')
        fig.tight_layout()
        fig.suptitle('Gibbs sampling')
        # plt.savefig('gibbs.pdf')
        plt.show()

        xrange = np.linspace(1.5,2.5,1000)
        slope_hist = rv_histogram(np.histogram(self.ws[:,0], bins=100))
        slope = xrange[slope_hist.pdf(xrange).argmax()]
        
        xrange = np.linspace(200,420,1000)
        intercept_hist = rv_histogram(np.histogram(self.ws[:,1], bins=69))
        intercept = xrange[intercept_hist.pdf(xrange).argmax()]
        return intercept, slope

if __name__ == '__main__':
    df = pd.read_csv('new_york_bicycles2.csv')
    X_whole = df.values
    X = np.column_stack([X_whole[:,0], np.ones(len(X_whole))])
    y = X_whole[:,1]

    lol = LinRegGibbs()
    lol.fit(X, y, n=10000)
    intercept, slope = lol.plot_result()

    xrange = np.linspace(0,5500,1000)
    fig, ax = plt.subplots(figsize=(7,5))
    ax.scatter(*X_whole.T)
    ax.plot(xrange, slope*xrange+intercept, c='red')
    ax.text(
        x=0.4, 
        y=0.85, 
        s=r'$w(x) = {a:.4f}x + {b:.4f}$'.format(a=slope, b=intercept), 
        horizontalalignment='center', 
        verticalalignment='center', 
        transform=ax.transAxes
    )
    fig.suptitle('Weights from Gibbs')
    # plt.savefig('regline_gibbs.pdf')
    plt.show()