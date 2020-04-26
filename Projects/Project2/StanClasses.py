from abc import abstractmethod
import pystan
import numpy as np 
import pandas as pd
import utils

class BaseMCMCFactorizer:
    '''
    Template class for classes for matrix decomposition
    '''
    @abstractmethod
    def fit(self, X: np.ndarray):
        pass

    def transform(self, X: np.ndarray):
        pass

    def inverse_transform(self, X):
        pass

class MF_normalnormal(BaseMCMCFactorizer):
    '''
    Class for probabilistic matrix decomposition using MCMC.
    '''
    def __init__(self, n_components: int=2, stanfile: str='sm_normalnormal.stan', 
                 cache_name: str='normalnormal'):
        self.n_components = n_components
        self.components_ = None

    def _init_fit(self, df):
        self.data_normalnormal = dict(
            n_components=self.n_components, n=df.shape[0], m=df.shape[1], p=p,
            q=q, X=X, mu_u=0, sigma_u=10, mu_v=0, sigma_v=10, sigma_x=1
        )
        
    def fit_transform(self, X: np.ndarray, user_column: str='user_id', ):
        '''
        X: DataFrame containing sparse data format
        '''
        X 

        return self

    def fit(self, *args, **kwargs):
        '''
        Invokes self.fit_transform(*args, **kwargs)
        and returns self
        '''
        self.fit_transform(*args, **kwargs)
        return self