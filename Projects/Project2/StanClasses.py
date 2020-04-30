from abc import abstractmethod
import pystan
import numpy as np
import pandas as pd
import utils
from matplotlib import pyplot as plt

class BaseStanFactorizer:
    '''
    Template class for classes for matrix decomposition
    '''
    def __init__(self):
        self.fitted = False

    @abstractmethod
    def fit_transform(self, X: np.ndarray):
        '''
        This abstract method should create the instance attributes "Us"
        and "Vs" and return them. It should also invoke the self.set_fitted 
        method
        '''
    
    def fit(self, *args, **kwargs):
        '''
        Invokes self.fit_transform(*args, **kwargs)
        and returns self
        '''
        self.fit_transform(*args, **kwargs)
        return self

    @abstractmethod
    def _predictive_sample(self, P: np.ndarray) -> np.ndarray:
        '''
        Function used for predictive sampling
        '''

    def assert_fitted(self):
        assert self.fitted, 'Model not fitted'

    def set_fitted(self):
        self.fitted = True

    def mae(self, df: pd.DataFrame):
        '''
        Computes mean absolute error

        df: should represent matrix in sparse format, each row should
            consists only of [row_index, col_index, value] in that order
        '''
        # Xs is a 3D array
        Xs = self.Us@self.Vs

        row_inds = df.iloc[:,0]
        col_inds = df.iloc[:,1]
        ratings = df.iloc[:,2].values

        y_preds = np.array([X[row_inds, col_inds] for X in Xs])
        abserrors = np.abs(y_preds - ratings)
        return abserrors.mean()

    def _plot_ci(self, n, P, lower_bounds, upper_bounds, ax, *args,
                 **kwargs):
        '''
        Plots credible intervals
        '''
        means = P.mean(axis=0)

        if ax is None:
            ax = plt.gca()

        ax.errorbar(range(n), means,
                    yerr=[means-lower_bounds, upper_bounds-means],
                    fmt='o', linewidth=2, *args, **kwargs)

    def ci(self, n_elements: int=20, n_samples: int=1000, p=0.95,
        show: bool=False, ax: 'matplotlib.Axes'=None, *args, **kwargs):
        '''
        Computes credible intervals first elements of matrix.

        Parameters
        ------------
        n_elements: Number of elements to calculte credible intervals for
        n_samples: Number of samples to sample from predictive distribution
        p: Optional, credible interval percentage, 0.95 by default
        show: Optional, to plot credible intervals or not, False by default
        ax: Optional, plots on given ax, no effect if show is False

        Returns
        --------
        (lower_bounds, upper_bounds)
        '''
        self.assert_fitted()

        # This is equivalent to Xs = np.array([U@V for U,V in zip(Us, Vs)])
        Xs = self.Us@self.Vs

        # Used to extract first n_elements from predicted Xs
        row_inds, col_inds = np.unravel_index(range(n_elements), Xs.shape[1:])

        # Sample from predictive distribution
        picks = np.random.randint(0, len(Xs), n_samples)
        P = Xs[picks][:,row_inds, col_inds]
        P = self._predictive_sample(P)
        P.sort(axis=0)

        # Get credible intervals of samples from predictive distribution
        half_p = (1-p)/2
        lb = np.floor((half_p*n_samples)).astype(int)
        ub = np.ceil((p+half_p)*n_samples).astype(int)

        lower_bounds, upper_bounds = P[lb], P[ub]

        if show:
            self._plot_ci(n_elements, P, lower_bounds, upper_bounds, ax, *args,
                        **kwargs)

        return lower_bounds, upper_bounds
    
class SimpleFactorizer(BaseStanFactorizer):
    '''
    Class for probabilistic matrix decomposition using MCMC.
    '''
    def __init__(self, n_components: int=2, mu_u: float=1, sigma_u: float=5,
                 mu_v: float=1, sigma_v=5, sigma_x=1,
                 stanfile: str='sm_simple.stan', cache_name: str='simple'):
        '''
        Factorization: X \approx UV, where X is the dense matrix

        Model:
        U ~ N(mu_u, sigma_u)
        V ~ N(mu_v, sigma_v)
        X ~ N(UV, sigma_x)

        Parameters
        -----------
        n_components: Embedding dimension
        mu_u: mean of elements in U
        sigma_u: std of elements in U
        mu_v: mean of elements in
        sigma_v: std of elements in V
        sigma_x: std of elements in X
        stanfile: file with stancode
        cache_name: name for compiled model
        '''
        super().__init__()
        self.stanfile = stanfile
        self.cache_name = cache_name

        self.n_components = n_components
        self.components_ = None

        self.mu_u = mu_u
        self.sigma_u = sigma_u

        self.mu_v = mu_v
        self.sigma_v = sigma_v

        self.sigma_x = sigma_x

    def _predictive_sample(self, P):
        self.assert_fitted()
        return np.random.normal(loc=P, scale=self.sigma_x, size=P.shape)

    def fit_transform(self, df: pd.DataFrame, **kwargs):
        '''
        Parameters
        -----------
        df: should represent matrix in sparse format, each row should
            consists only of [row_index, col_index, value] in that order

        kwargs: keyword arguments for StanModel.sampling()

        Returns
        --------
        Sampled values
        (Us, Vs)
        '''
        # Get shape (p, q) of dense matrix
        self.p = len(df.iloc[:,0].unique())
        self.q = len(df.iloc[:,1].unique())

        datadict = dict(
            n_components=self.n_components, n=len(df), p=self.p,
            q=self.q, df=df, mu_u=self.mu_u, sigma_u=self.sigma_u,
            mu_v=self.mu_v, sigma_v=self.sigma_v, sigma_x=self.sigma_x
        )

        self.code = utils.get_stan_code(self.stanfile)
        self.sm = utils.StanModel_cache(self.code, model_name=self.cache_name)

        self.stanfit = self.sm.sampling(datadict, **kwargs)
        self.Us, self.Vs, self.lp__ =\
            self.stanfit['U'], self.stanfit['V'], self.stanfit['lp__']

        # Set fitted flag
        self.set_fitted()

        return self.Us, self.Vs

class NonNegativeFactorizer(BaseStanFactorizer):
    '''
    Class for probabilistic matrix decomposition using MCMC.
    '''
    def __init__(self, n_components: int=2, mu_u: float=1, sigma_u: float=5,
                 mu_v: float=1, sigma_v=5, sigma_x=1,
                 stanfile: str='sm_nmf.stan', cache_name: str='nmf'):
        '''
        Factorization: X \approx UV, where X is the dense matrix

        Model:
        U ~ N(mu_u, sigma_u)
        V ~ N(mu_v, sigma_v)
        X ~ N(UV, sigma_x)

        Parameters
        -----------
        n_components: Embedding dimension
        mu_u: mean of elements in U
        sigma_u: std of elements in U
        mu_v: mean of elements in
        sigma_v: std of elements in V
        sigma_x: std of elements in X
        stanfile: file with stancode
        cache_name: name for compiled model
        '''
        super().__init__()
        self.stanfile = stanfile
        self.cache_name = cache_name

        self.n_components = n_components
        self.components_ = None

        self.mu_u = mu_u
        self.sigma_u = sigma_u

        self.mu_v = mu_v
        self.sigma_v = sigma_v

        self.sigma_x = sigma_x

    def _predictive_sample(self, P):
        self.assert_fitted()
        return np.random.normal(loc=P, scale=self.sigma_x, size=P.shape)

    def fit_transform(self, df: pd.DataFrame, **kwargs):
        '''
        Parameters
        -----------
        df: should represent matrix in sparse format, each row should
            consists only of [row_index, col_index, value] in that order

        kwargs: keyword arguments for StanModel.sampling()

        Returns
        --------
        Sampled values
        (Us, Vs)
        '''
        # Get shape (p, q) of dense matrix
        self.p = len(df.iloc[:,0].unique())
        self.q = len(df.iloc[:,1].unique())

        datadict = dict(
            n_components=self.n_components, n=len(df), p=self.p,
            q=self.q, df=df, mu_u=self.mu_u, sigma_u=self.sigma_u,
            mu_v=self.mu_v, sigma_v=self.sigma_v, sigma_x=self.sigma_x
        )

        self.code = utils.get_stan_code(self.stanfile)
        self.sm = utils.StanModel_cache(self.code, model_name=self.cache_name)

        self.stanfit = self.sm.sampling(datadict, **kwargs)
        self.Us, self.Vs, self.lp__ =\
            self.stanfit['U'], self.stanfit['V'], self.stanfit['lp__']

        # Set fitted flag
        self.set_fitted()

        return self.Us, self.Vs

class ARD_Factorizer(BaseStanFactorizer):
    '''
    Class for probabilistic matrix decomposition using MCMC.
    '''
    def __init__(self, n_components: int=2, mu_u: float=1, sigma_u: float=5,
                 mu_v: float=1, sigma_v=5, sigma_x=1,
                 stanfile: str='sm_simple.stan', cache_name: str='simple'):
        '''
        Factorization: X \approx UV, where X is the dense matrix

        Model:
        U ~ N(mu_u, sigma_u)
        V ~ N(mu_v, sigma_v)
        X ~ N(UV, sigma_x)

        Parameters
        -----------
        n_components: Embedding dimension
        mu_u: mean of elements in U
        sigma_u: std of elements in U
        mu_v: mean of elements in
        sigma_v: std of elements in V
        sigma_x: std of elements in X
        stanfile: file with stancode
        cache_name: name for compiled model
        '''
        super().__init__()
        self.stanfile = stanfile
        self.cache_name = cache_name

        self.n_components = n_components
        self.components_ = None

        self.mu_u = mu_u
        self.sigma_u = sigma_u

        self.mu_v = mu_v
        self.sigma_v = sigma_v

        self.sigma_x = sigma_x

    def _predictive_sample(self, P):
        self.assert_fitted()
        return np.random.normal(loc=P, scale=self.sigma_x, size=P.shape)

    def fit_transform(self, df: pd.DataFrame, **kwargs):
        '''
        Parameters
        -----------
        df: should represent matrix in sparse format, each row should
            consists only of [row_index, col_index, value] in that order

        kwargs: keyword arguments for StanModel.sampling()

        Returns
        --------
        Sampled values
        (Us, Vs)
        '''
        # Get shape (p, q) of dense matrix
        self.p = len(df.iloc[:,0].unique())
        self.q = len(df.iloc[:,1].unique())

        datadict = dict(
            n_components=self.n_components, n=len(df), p=self.p,
            q=self.q, df=df, mu_u=self.mu_u, sigma_u=self.sigma_u,
            mu_v=self.mu_v, sigma_v=self.sigma_v, sigma_x=self.sigma_x
        )

        self.code = utils.get_stan_code(self.stanfile)
        self.sm = utils.StanModel_cache(self.code, model_name=self.cache_name)

        self.stanfit = self.sm.sampling(datadict, **kwargs)
        self.Us, self.Vs, self.lp__ =\
            self.stanfit['U'], self.stanfit['V'], self.stanfit['lp__']

        # Set fitted flag
        self.set_fitted()

        return self.Us, self.Vs

