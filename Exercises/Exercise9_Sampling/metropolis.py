import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 
from scipy.stats import multivariate_normal, gamma, norm, rv_histogram
np.random.seed(42069)

class LinRegMetropolis:
    def __init__(self, slope=1, intercept=1, beta=1, 
                 alpha=10, beta_=0.1, a=1, b=1e-2):
        self.a = a 
        self.b = b 
        self.alpha = alpha 
        self.beta_ = beta_ # For posteriror 
        self.init_slope = slope
        self.init_intercept = intercept
        self.init_beta = beta

    def fit(self, X, y, n=69000):
        self.n = n
        self.X = X 
        self.y = y 
        self.thetas = np.empty((n,3))
        self.thetas[0] = np.array([self.init_slope, self.init_intercept, self.init_beta])
        self.lhoods = np.empty(n)
        self.lhoods[0] = self.posterior(self.thetas[0])

        sigma = np.eye(3)*0.69
        updates = 0
        for i in range(1,n):
            proposal = multivariate_normal.rvs(mean=self.thetas[i-1], cov=sigma)
            lhood_prop = self.posterior(proposal)
            ratio = lhood_prop/self.lhoods[i-1]
            A = min((1,ratio))
            if np.random.uniform(0,1,1) < A:
                updates += 1
                self.thetas[i] = proposal
                self.lhoods[i] = lhood_prop
            else:
                self.thetas[i] = self.thetas[i-1]
                self.lhoods[i] = self.lhoods[i-1]
        return updates

    def posterior(self, theta):
        epsilon = 1e-16
        I = np.eye(2)
        origo = np.zeros(2)

        # First two elements are for linreg weights
        w = theta[:2]
        beta = theta[-1]

        logw = multivariate_normal.logpdf(w, origo, self.alpha*I).clip(epsilon)
        logbeta = gamma.logpdf(beta, self.a, self.b).clip(epsilon)    
        logdata =\
            norm.logpdf(x=self.y-w@X.T, loc=0, scale=1/(beta+epsilon)).clip(epsilon).sum()
    
        return logw+logbeta+logdata

    def plot_result(self):
        fig, axes = plt.subplots(4,2, figsize=(7,6))
        xrange = np.arange(2,self.n+1)
        ax_left = axes[:,0].ravel()
        ax_right = axes[:,1].ravel()
        ax_left[0].plot(xrange, self.lhoods[1:], linewidth=0.69)
        ax_left[0].set_title(r'Posterior trace')
        ax_left[1].plot(xrange, self.thetas.T[0][1:], linewidth=0.69)
        ax_left[1].set_title(r'$w_1^t$ (slope) trace')
        ax_left[2].plot(xrange, self.thetas.T[1][1:], linewidth=0.69)
        ax_left[2].set_title(r'$w_0^t$ (intercept) trace')
        ax_left[3].plot(xrange, self.thetas.T[2][1:], linewidth=0.69)
        ax_left[3].set_title(r'$\beta^t$ trace')
        
        ax_right[0].hist(self.lhoods[1:], bins=40)
        ax_right[0].set_title(r'Posterior hist')
        ax_right[1].hist(self.thetas.T[0][1:], bins=40)
        ax_right[1].set_title(r'$w_1^t$ hist')
        ax_right[2].hist(self.thetas.T[1][1:], bins=40)
        ax_right[2].set_title(r'$w_0^t$ hist')
        ax_right[3].hist(self.thetas.T[2][1:], bins=40)
        ax_right[3].set_title(r'$\beta^t$ hist')
        fig.suptitle('Metropolis')
        fig.tight_layout()
        plt.savefig('metropolis2.pdf')
        plt.show()

        print('Enter burnout cutoff:')
        burnout = int(input())
        xrange = np.linspace(1,10,10000)
        slope_hist = rv_histogram(np.histogram(self.thetas.T[0][burnout:], bins=100))
        slope = xrange[slope_hist.pdf(xrange).argmax()]
        
        xrange = np.linspace(1,5000,10000)
        intercept_hist = rv_histogram(np.histogram(self.thetas.T[1][burnout:], bins=100))
        intercept = xrange[intercept_hist.pdf(xrange).argmax()]
        return intercept, slope

if __name__ == '__main__':
    df = pd.read_csv('new_york_bicycles2.csv')
    X_whole = df.values
    X = np.column_stack([X_whole[:,0], np.ones(len(X_whole))])
    y = X_whole[:,1]

    from tqdm import tqdm
    # import warnings
    # warnings.filterwarnings('ignore')

    lol = LinRegMetropolis()    
    lol.fit(X,y) 
    intercept, slope = lol.plot_result()

    xrange = np.linspace(0,5500,1000)
    fig, ax = plt.subplots(figsize=(7,5))
    fig.suptitle('Weights from Metropolis')
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
    plt.savefig('regline_metropolis.pdf')
    plt.show()
