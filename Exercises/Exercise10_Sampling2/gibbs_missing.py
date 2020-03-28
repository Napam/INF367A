import numpy as np 
import pandas as pd 
from scipy.stats import norm, multivariate_normal
from matplotlib import pyplot as plt
# np.random.seed(69)

class GibbsSampler:
    def __init__(self, n_gibbs: int=2000):
        self.n_gibbs = n_gibbs

    def _init_fit(self, X):
        self.X = X 
        self.n = len(self.X)
        self.miss_mask = np.isnan(self.X)
        self.n_missing = self.miss_mask.sum()

        # To be used as intermediate placeholder for Z values 
        self.XZ = self.X.copy()

        self.missing_indices = np.where(self.miss_mask[:,0])[0]

        self.mus = np.empty((self.n_gibbs, 2))
        self.zs = np.empty((self.n_gibbs, self.n_missing))

        rho = 0.8
        self.Sigma = np.array([[1,rho],[rho,1]])

        self.mus[0] = [0,0]
        self.zs[0] = 1
        self.XZ[self.miss_mask] = 1 

    def _sample_mu(self, i: int):
        return multivariate_normal.rvs(self.XZ.mean(0), self.Sigma)
    
    def _sample_z(self, i: int):
        return norm.rvs(self.mus[i,0], self.Sigma[0,0], size=self.n_missing)

    def _determine_z(self):
        self.z_hat = np.empty(self.n_missing)
        for i, z in enumerate(self.zs.T):
            bins, edges = np.histogram(z, bins='auto')
            self.z_hat[i] = edges[bins.argmax()+1]
        self.XZ[self.miss_mask] = self.z_hat

    def _determine_mu(self):
        self.mu_hat = np.empty(2)
        for i, mu in enumerate(self.mus.T):
            bins, edges = np.histogram(mu, bins='auto')
            self.mu_hat[i] = edges[bins.argmax()+1]

    def fit(self, X: np.ndarray):
        self._init_fit(X)

        for i in range(1,self.n_gibbs):
            self.mus[i]=self._sample_mu(i-1)
            self.zs[i] = self._sample_z(i)
            self.XZ[self.miss_mask]=self.zs[i]
        
        self._determine_mu()
        self._determine_z()

    def plot_result(self):
        colors = self.miss_mask[:,0]
        fig, axes = plt.subplots(3,2)

        axes[0,0].plot(self.mus[:,0])
        axes[1,0].plot(self.mus[:,1])
        axes[2,0].plot(self.zs[:,4])
        axes[0,1].hist(self.mus[:,0])
        axes[1,1].hist(self.mus[:,1])
        axes[2,1].hist(self.zs[:,4])
        plt.show()

        plt.scatter(*self.XZ.T, c=colors)
        plt.scatter(*self.X.T)
        plt.scatter(*self.mu_hat)
        plt.show()

if __name__ == '__main__':
    df = pd.read_csv('missing_data.csv', header=None)
    # plt.scatter(*df.values.T)
    # plt.show()

    LadyGaga = GibbsSampler()
    LadyGaga.fit(df.values)
    LadyGaga.plot_result()

    