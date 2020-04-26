import numpy as np 
from matplotlib import pyplot as plt 
from EM_GMM import GMM
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from typing import Union
from matplotlib.patches import Ellipse
np.random.seed(69)

class GMM2(GMM):
    def _prepare_before_fit(self, X: np.ndarray) -> None:
        '''
        Prepares object attributes and such before fitting
        '''
        self.X = X
        self.N, self.dim = X.shape
        self.X_std = np.std(X)

        if self.init_covariance == 'auto':
            self.init_covariance = np.var(X)

        # Initialize component placeholders
        self.components = np.empty(self.k, dtype=self._init_dtype(self.dim))

        # Pick random points as initial mean positions
        # Reshape to handle case for 1-dim data
        self.components['mean'] = \
            X[np.random.choice(range(self.N), self.k, replace=False)].reshape(*self.components['mean'].shape)
        # Initialize covariance matrices with scaled identity matrices
        # self.components['cov'] = np.repeat(self.init_covariance*np.eye(self.dim)[np.newaxis,...], self.k, axis=0) 
        self.components[0]['cov'] = np.array([[1,0.8],[0.8,1]])
        # Initialize uniform mixing weights
        self.components['mix'] = np.full(self.k, 1/self.k)

        # Weight for each data point, columns are respective to components
        self.weights = np.empty((self.N, self.k))  

        self.hood_history = []
        # Calculate starting weights in order to calculate initial likelihood
        self._E_step() # This automatically logs likelihood

        #EM iterations
        self.em_iterations = 0 

    def _E_step(self):
        print(self.components)
        lhood = np.log(mvn.pdf(self.X, self.components[0]['mean'], self.components[0]['cov'])).sum()
        self.hood_history.append(lhood)
        return lhood
    
    def _M_step(self):
        self.X[1,0] = self.components[0]['mean'][0]+0.8*(self.X[1][1]-self.components[0]['mean'][0])
        x1 = self.X[0]
        x2 = self.X[1]
        x3 = self.X[2]

        self.components[0]['mean'] = (x1+x2+x3)/3

if __name__ == '__main__':
    X = np.array([
        [1.79, 4.11],
        [42069, 5.16],
        [2.9, 3.9],
    ])

    lol = GMM2(k=1)
    lol.fit(X, atol=1e-10)
    lol.plot_result()