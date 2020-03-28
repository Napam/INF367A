import pandas as pd 
import numpy as np 
from scipy.stats import gamma, poisson, sem, bayes_mvs
from matplotlib import pyplot as plt 

np.random.seed(69)

def HDI_from_MCMC(posterior_samples, credible_mass):
        # https://stackoverflow.com/questions/22284502/highest-posterior-density-region-and-central-credible-region
        # Computes highest density interval from a sample of representative values,
        # estimated as the shortest credible interval
        # Takes Arguments posterior_samples (samples from posterior) and credible mass (normally .95)
        sorted_points = sorted(posterior_samples)
        ciIdxInc = np.ceil(credible_mass * len(sorted_points)).astype('int')
        nCIs = len(sorted_points) - ciIdxInc
        
        ciWidth = [None]*nCIs
        for i in range(nCIs):
            ciWidth[i] = sorted_points[i + ciIdxInc] - sorted_points[i]

        HDImin = sorted_points[ciWidth.index(min(ciWidth))]
        HDImax = sorted_points[ciWidth.index(min(ciWidth))+ciIdxInc]
        return (HDImin, HDImax)

def maxima(ax, x, y):
    best_idx = np.argmax(y)
    bestx = x[best_idx]
    besty = y[best_idx]

    ax.scatter(bestx, besty, color='k', marker='x',zorder=10)
    ax.text(bestx, besty + 0.01, f'({bestx:.2f}, {besty:.2f})')

if __name__ == '__main__':
    def task1b(show=True):
        '''Compute posterior'''
        df = pd.read_csv('exercise5_1.txt', header=None)
        X = df.values.ravel()

        a0 = 2
        b0 = 2
        n = len(X)

        a = a0 + X.sum()
        b = n + b0

        xrange = np.linspace(0, 5, 10000)

        # Prior
        y_before = gamma.pdf(xrange, a=a0, scale=1/b0)

        # Posterior
        y_after = gamma.pdf(xrange, a=a, scale=1/b)

        # print(y)

        lambda_approx = xrange[np.argmax(y_after)]

        if show:
            fig, ax = plt.subplots(figsize=(7,4))
            fig.suptitle('Before and after observing data')
            ax.plot(xrange, y_before, label='Prior')
            ax.plot(xrange, y_after, label='Posterior')
            ax.set_xlabel(r'$\lambda$')
            ax.set_ylabel('density')

            ax.legend()
            maxima(ax, xrange, y_after)
            maxima(ax, xrange, y_before)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            # plt.savefig('images/1b.pdf')
            plt.show()

            print(lambda_approx)

        return lambda_approx

    def task1c(show=True):
        '''MC sanity check'''
        df = pd.read_csv('exercise5_1.txt', header=None)
        X = df.values.ravel()

        lambda_approx = task1b(False)
        y = poisson.rvs(lambda_approx, size=100)
        
        if show:
            histkwargs = dict(edgecolor='black', density=True)
            fig, (ax1, ax2) = plt.subplots(1,2, sharex=True, figsize=(7,3))
            fig.suptitle('Comparing data to monte carlo sampling')
            ax1.set_title('Data', fontsize=10)
            ax1.hist(X, **histkwargs)
            ax2.set_title('Sampled data from posterior estimate', fontsize=10)
            ax2.hist(y, **histkwargs)

            plt.setp([ax1, ax2], xlabel='Poisson frequencies', ylabel='Normalized frequencies')

            fig.tight_layout(rect=[0, 0.03, 1, 0.90])
            # plt.savefig('images/1c.pdf')
            plt.show()
        
        return y

    def task1d(show=True):
        '''Credibility interval'''
        y = task1c(False)
    
        stringy = f'95% credibility interval is ()'
        def print_ci(y: np.ndarray, cm: float):
            left, right = HDI_from_MCMC(y, cm)
            print(f'{cm*100:.0f}% Credibility interval: ({left}, {right})')

        print_ci(y, 0.5)
        print_ci(y, 0.95)

    
    # task1b()
    # task1c(True)
    # task1d()