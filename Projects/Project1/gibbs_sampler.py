import numpy as np
np.random.seed(42069)
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import gamma, norm
from scipy.stats import multivariate_normal as mvn
from typing import Callable, Union
from mpl_toolkits.mplot3d import Axes3D

'''
LinearRegressionARD is the one implementing ARD
It subclasses the LinearRegression class. 
'''

class LinearRegression:
    '''
    Linear regression using Gibbs Sampling without ARD
    '''
    def __init__(self, init_ws: np.ndarray=None, init_beta: float=None,
                 polydegree: int=1, alpha: float=1, a: float=1, 
                 b: float=1, n_gibbs: int=1024):
        '''
        polydegree: degree of polynomial kernel

        init_ws: inital regression model parameters
        init_beta: initial noise precision

        if any of the initial parameteres are None, they will simply be
        sampled from Uniform(0.2,2)

        polydegree: Degree of polynomial regression

        a: hyperparameter for P(beta|a,b)
        b: hyperparameter for P(beta|a,b)

        n_gibbs: Number of samples to sample when fitting using Gibbs
                 sampling
        '''
        self.init_ws = init_ws
        self.init_beta = init_beta

        self.polydegree=polydegree

        self.a=a
        self.b=b
        self.alpha = alpha

        self.n_gibbs = n_gibbs

    def _sample_w(self, i: int) -> np.ndarray:
        '''
        Sample a set of model parameters w from its full conditional
        distribution P(w|y,x,alpha,beta)

        i: iteration index
        '''
        # self.D_cov is self.alpha * I
        S = np.linalg.inv(self.D_cov + self.betas[i]*self.X.T@self.X)
        m = self.betas[i]*np.linalg.multi_dot((S,self.X.T,self.y))    
        return mvn.rvs(m,S)

    def _sample_beta(self, i: int) -> float:
        '''
        Sample a set of prior values beta from its full conditional
        distribution P(beta|y,x,w,alpha)

        i: iteration index
        '''
        p = self.y - self.ws[i]@self.X.T
        return gamma.rvs(a=self.a+self.n/2, scale=1/(self.b+0.5*p.T@p))

    def _logposterior(self, i: int):
        '''
        Calculate posterior

        i: iteration index
        '''
        origo = np.zeros(self.dim)

        # self.D_cov is 1/self.alpha * I
        log_w =\
        mvn.logpdf(self.ws[i], mean=origo, cov=self.D_cov).sum()

        log_beta =\
        gamma.logpdf(self.betas[i], a=self.a, scale=self.b).sum()

        log_data =\
        norm.logpdf(self.y, loc=self.ws[i]@self.X.T, 
                    scale=np.sqrt(1/self.betas[i])).sum()
            
        return log_w + log_beta + log_data

    def _init_fit(self, X, y):
        '''
        Setup before fitting
        '''
        assert len(X) == len(y), 'Length mismatch between X and y'
        assert len(X.shape) == 2, 'X must be 2D array, try X.reshape(-1,1)'

        self.PolyTransformer=PolynomialFeatures(
            degree=self.polydegree,
            include_bias=True
        )

        self.X = self.PolyTransformer.fit_transform(X)
        self.y = y

        # Number of data points
        self.n = len(y)
        # Dimension of data / number of model weights
        self.dim = self.X.shape[1]

        # Initalize memory for w, beta, alpha
        self.ws = np.empty((self.n_gibbs, self.dim))     # vectors
        self.betas = np.empty(self.n_gibbs)              # scalars
        self.logposteriors = np.empty(self.n_gibbs)      # scalars

        # Initial model parameters
        if self.init_ws is None:
            self.init_ws = np.random.normal(
                loc=0, scale=self.alpha, size=self.dim)
        if self.init_beta is None:
            self.init_beta = np.random.gamma(
                shape=self.a, scale=self.b, size=1)

        # To be used in self._sample_w
        self.D_cov = np.eye(self.dim)*self.alpha

        self.ws[0] = self.init_ws
        self.betas[0] = self.init_beta
        self.logposteriors[0] = self._logposterior(0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegressionARD':
        '''
        Fit model using Gibbs sampling

        X: Predictors
        y: Response
        '''
        self._init_fit(X, y)

        for i in range(1, self.n_gibbs):
            self.ws[i] = self._sample_w(i-1)
            self.betas[i] = self._sample_beta(i)
            self.logposteriors[i] = self._logposterior(i)

        self._determine_w()
        self._determine_beta()
        return self

    def _determine_w(self):
        '''
        Determines w_hat by taking argmax of histogram
        '''
        self.w_hat = np.empty(self.dim)
        for i, w in enumerate(self.ws.T):
            hist, edges = np.histogram(w[self.n_gibbs//4:], bins='auto')
            self.w_hat[i]=edges[hist.argmax()+1]
        return self.w_hat

    def _determine_beta(self):
        '''
        Determines beta_hat by taking argmax of histogram
        '''
        hist, edges = np.histogram(self.betas[self.n_gibbs//4:], bins='auto')
        self.beta_hat = edges[hist.argmax()+1]
        return self.beta_hat

    def get_w_hat(self, variable_names: Union[list, None]=None, 
                  show: bool=False):
        '''
        Get linear regression weights
        '''
        if show: 
            names = self.PolyTransformer.get_feature_names(variable_names)
            stringy=''
            for factor, coef in zip(names[::-1], self.w_hat[::-1]):
                if coef < 0:
                    stringy += f' {coef:.2f}{factor}'
                else:
                    stringy += f' +{coef:.2f}{factor}'
            print(stringy)
        return self.w_hat

    def get_mae(self) -> float:
        '''Returns mean absolute error of data'''
        residuals = self.y - self.w_hat@self.X.T
        mae_wrt_ws = np.mean(abs(residuals), axis=0)
        return mae_wrt_ws

    def _init_plot(self, figsize, selected_axes, skip):
        '''
        Initialize plot attributes
        '''
        self.fig, self.axes = plt.subplots(2+len(selected_axes),
                                           2, figsize=figsize)
        self.axes_traces, self.axes_hists = self.axes[:,0], self.axes[:,1]
        self.plot_dict = {
            'Log-posterior':self.logposteriors[skip:],
            r'$\beta$':self.betas[skip:],
        }

        for i in selected_axes:
            self.plot_dict[rf'$w_{{{i}}}$'] = self.ws[skip:,i]

        self.plot_kwargs = {'linewidth':0.2}
        self.hist_kwargs = {'bins':'auto', 'edgecolor':'k', 'linewidth':0.5}

    def plot_result(self, figsize: tuple=(5,5), axes: list=[0,1],
                    skip: int=10, show: bool=True, figtitle: str=None,
                    filename: str=None) -> None:
        '''
        Visualize results.

        The plots discards the datapoints from the first
        two iterations as they really mess up the plots. That is because
        they are outliers with vastly different values from the rest, but
        pyplot will still scale the plots such that they are included.

        If filename is given, the plot will be saved as filename (must 
        include filetype)
        '''
        self._init_plot(figsize, axes, skip)
        for i, title, data in zip(range(len(self.axes)),
                                 self.plot_dict.keys(),
                                 self.plot_dict.values()):
            self.axes_traces[i].set_title(title+' trace')

            self.axes_traces[i].plot(data, **self.plot_kwargs)
            self.axes_traces[i].axvline(len(data)//4, c='red')

            self.axes_hists[i].set_title(title+' hist')
            self.axes_hists[i].hist(data[len(data)//4:], **self.hist_kwargs)
        self.fig.tight_layout(rect=[0,0.03,1,0.95])

        if figtitle is not None: self.fig.suptitle(figtitle)
        if filename is not None: plt.savefig(filename)
        plt.close()
        if show: plt.show()
    
    def plot_w_traces(self, figsize: tuple=(5,5), axes: list=[0,1], 
                     n_cols: int=4, rowsize: int=1.5, colsize: int=2,
                     skip: int=10, show: bool=True, ticks: bool=False, 
                     figtitle: str=None, filename: str=None) -> None:
        '''
        Made to visualize many traceplots for weights.
        '''
        n = len(axes)
        n_cols=n if n < n_cols else n_cols
        n_rows=int(np.ceil(n/n_cols))
        fig, axes_ = plt.subplots(n_rows, n_cols, 
                        figsize=(colsize*n_cols,rowsize*n_rows))
        axes_ = np.ravel(axes_)

        plot_dict = {rf'$w_{{{i}}}$':self.ws[skip:,i] for i in axes}

        plot_kwargs = {'linewidth':0.2}
        for ax, title, data in zip(axes_, plot_dict.keys(), 
                                   plot_dict.values()):
            ax.set_title(title)
            ax.plot(data, **plot_kwargs)
            ax.axvline(len(data)//4, c='red')
        
        fig.tight_layout(rect=[0,0.03,1,0.95])

        # Remove ticks
        if not ticks: plt.setp(axes_, xticks=[], yticks=[])

        # Turn off spines on empty plots
        if n < n_cols*n_rows:
            [ax.axis('off') for ax in axes[-(n_cols*n_rows-n):]]

        if filename is not None: plt.savefig(filename)
        if show: 
            plt.show()
        else:
            return fig, axes

class LinearRegressionARD(LinearRegression):
    '''
    Linear regression with automatic relevance determination.
    Fitted using Gibbs sampling.
    '''
    def __init__(self, init_ws: np.ndarray=None, init_beta: float=None,
                 init_alphas:np.ndarray=None, polydegree: int=1, a: float=1,
                 b: float=1, c: float=1, d: float=1, 
                 n_gibbs: int=1024):
        '''
        polydegree: degree of polynomial kernel

        init_ws: inital regression model parameters
        init_beta: initial noise precision
        init_alphas: initial ARD precision weights

        if any of the initial parameteres are None, they will simply be
        sampled from Uniform(0.2,2)

        polydegree: Degree of polynomial regression

        a: hyperparameter for P(beta|a,b)
        b: hyperparameter for P(beta|a,b)

        c: hyperparameter for P(alpha|c,d)
        d: hyperparmater for P(alpha|c,d)

        n_gibbs: Number of samples to sample when fitting using Gibbs
                 sampling
        '''
        self.init_ws = init_ws
        self.init_beta = init_beta
        self.init_alphas = init_alphas

        self.polydegree=polydegree

        self.a=a
        self.b=b
        self.c=c
        self.d=d

        self.n_gibbs = n_gibbs

    def _sample_w(self, i: int) -> np.ndarray:
        '''
        Sample a set of model parameters w from its full conditional
        distribution P(w|y,x,alpha,beta)

        i: iteration index
        '''
        S = np.linalg.inv(self.betas[i]*self.X.T@self.X + self.D)
        m = self.betas[i]*S@(self.X.T@self.y)
        return mvn.rvs(mean=m, cov=S)

    def _sample_alpha(self, i: int) -> np.ndarray:
        '''
        Sample a set of model parameters alpha from its full conditional
        distribution P(alpha|y,x,w,beta)

        i: iteration index
        '''
        # gamma.rvs will return a vector
        alphas = gamma.rvs(a=self.c_, scale=1/((self.ws[i]**2)*0.5+self.d),
                          size=self.dim)
        # Update covariance 
        # np.fill_diagonal(self.D, 1/alphas)
        np.fill_diagonal(self.D, alphas)
        return alphas

    def _logposterior(self, i: int):
        '''
        Calculate posterior

        i: iteration index
        '''
        log_w =\
        norm.logpdf(self.ws[i], loc=0, scale=np.sqrt(1/self.alphas[i])).sum()

        log_alphas =\
        gamma.logpdf(self.alphas[i], a=self.c, scale=1/self.d).sum()

        log_betas =\
        gamma.logpdf(self.betas[i], a=self.a, scale=1/self.b).sum()

        log_data =\
        norm.logpdf(self.y, loc=self.ws[i]@self.X.T, 
                    scale=np.sqrt(1/self.betas[i])).sum()
        return log_w + log_alphas + log_betas + log_data

    def _init_fit(self, X, y):
        '''
        Setup before fitting
        '''
        assert len(X) == len(y), 'Length mismatch between X and y'
        assert len(X.shape) == 2, 'X must be 2D array, try X.reshape(-1,1)'

        self.PolyTransformer = PolynomialFeatures(
            degree=self.polydegree,
            include_bias=True
        )
        self.X = self.PolyTransformer.fit_transform(X)
        self.y = y

        # Number of data points
        self.n = len(y)
        # Dimension of data / number of model weights
        self.dim = self.X.shape[1]

        # Initalize memory for w, beta, alpha
        self.ws = np.empty((self.n_gibbs, self.dim))     # vectors
        self.betas = np.empty(self.n_gibbs)              # scalars
        self.alphas = np.empty((self.n_gibbs, self.dim)) # vectors
        self.logposteriors = np.empty(self.n_gibbs)      # scalars

        # Initial model parameters
        if self.init_alphas is None:
            self.init_alphas = np.random.gamma(
                shape=self.c, scale=self.d, size=self.dim)
        if self.init_ws is None:
            self.init_ws = np.random.normal(
                loc=0, scale=self.init_alphas, size=self.dim)
        if self.init_beta is None:
            self.init_beta = np.random.gamma(
                shape=self.a, scale=self.b, size=1)

        self.ws[0] = self.init_ws
        self.betas[0] = self.init_beta
        self.alphas[0] = self.init_alphas
        self.logposteriors[0] = self._logposterior(0)

        # To be used in self._sample_w() and will be updated in place
        # in self._sample_alpha()
        self.D = np.eye(self.dim)
        np.fill_diagonal(self.D, self.init_alphas)

        # To be used in self._sample_alpha()
        self.c_ = np.full(self.dim, self.c+0.5)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegressionARD':
        '''
        Fit model using Gibbs sampling

        X: Predictors
        y: Response
        '''
        self._init_fit(X, y)

        for i in range(1, self.n_gibbs):
            self.ws[i] = self._sample_w(i-1)
            self.alphas[i] = self._sample_alpha(i)
            self.betas[i] = self._sample_beta(i)
            self.logposteriors[i] = self._logposterior(i)

        self._determine_w()
        self._determine_alpha()
        self._determine_beta()
        return self

    def _determine_alpha(self):
        '''
        Determine finalized values for alpha by taking argmax of 
        histogram after cutoff 
        '''
        self.alpha_hat = np.empty(self.dim)
        for i, w in enumerate(self.ws.T):
            hist, edges = np.histogram(w[self.n_gibbs//4], bins='auto')
            self.alpha_hat[i]=edges[hist.argmax()+1]
        return self.alpha_hat

    def _init_plot(self, figsize, selected_axes, skip):
        '''
        Initialize plot attributes
        '''
        self.fig, self.axes = plt.subplots(2+len(selected_axes)*2,
                                           2, figsize=figsize)
        self.axes_traces, self.axes_hists = self.axes[:,0], self.axes[:,1]
        self.plot_dict = {
            'Log-posterior':self.logposteriors[skip:],
            r'$\beta$':self.betas[skip:],
        }

        for i in selected_axes:
            self.plot_dict[rf'$w_{i}$'] = self.ws[skip:,i]
            self.plot_dict[rf'$\alpha_{i}$'] = self.alphas[skip:,i]

        self.plot_kwargs = {'linewidth':0.69}
        # Histogram bin count calculated using Scott's rule
        self.hist_kwargs = {'bins':'auto', 'edgecolor':'k', 'linewidth':0.69}

    def plot_alpha_traces(self, figsize: tuple=(5,5), axes: list=[0,1], 
                     n_cols: int=4, rowsize: int=1.5, colsize: int=2,
                     skip: int=10, show: bool=True, ticks: bool=False, 
                     figtitle: str=None, filename: str=None) -> None:
        '''
        Made to visualize many traceplots for weights.
        '''
        n = len(axes)
        n_cols=n if n < n_cols else n_cols
        n_rows=int(np.ceil(n/n_cols))
        fig, axes_ = plt.subplots(n_rows, n_cols, 
                        figsize=(colsize*n_cols,rowsize*n_rows))
        axes_ = np.ravel(axes_)

        plot_dict = {rf'$\alpha_{{{i}}}$':self.alphas[skip:,i] for i in axes}

        plot_kwargs = {'linewidth':0.2}
        for ax, title, data in zip(axes_, plot_dict.keys(), 
                                   plot_dict.values()):
            ax.set_title(title)
            ax.plot(data, **plot_kwargs)
            ax.axvline(len(data)//4, c='red')
        
        fig.tight_layout(rect=[0,0.03,1,0.95])

        # Remove ticks
        if not ticks: plt.setp(axes_, xticks=[], yticks=[])

        # Turn off spines on empty plots
        if n < n_cols*n_rows:
            [ax.axis('off') for ax in axes[-(n_cols*n_rows-n):]]

        if filename is not None: plt.savefig(filename)
        if show: 
            plt.show()
        else:
            return fig, axes

class PolyFunc:
    '''
    Class to generate polynomial dummy data for regression tasks.

    I want to have the nice string form of the functions
    so I can use them as latex plot titles later so I made
    this class.

    (To be honest I just wanted to apply my newfound RegEx knowledge)
    '''
    def __init__(self, strfunc: str):
        '''
        strfunc: Multivariate polynomial written in LaTeX syntax.
        '''
        self.strfunc = strfunc
        self.python_strfunc = self._parse_input(strfunc)
        self.signature = self._get_lambda_signature(self.python_strfunc)
        self.f = eval(self.signature+self.python_strfunc)
        self.X = None

    def _get_lambda_signature(self, stringybingy) -> str:
        '''
        Returns for example lambda: x,y
        '''
        # Find all varables
        temp = re.findall('[a-zA-Z]', stringybingy)
        # Get unique variables and sort them alphabetically
        temp = sorted(list(set(temp)))
        self.vars = temp.copy()
        self.n_vars = len(temp)

        # Construct lambda signature
        signature = 'lambda '
        for var in temp:
            signature += var+','
        signature += ': '
        return signature

    @staticmethod
    def _parse_input(stringybingy: str) -> str:
        '''
        LaTeX polynomial to Python parser
        '''
        # Replace spaces and stuff with nothing
        temp = re.sub('\s+','',stringybingy)
        # 420x -> 420*x
        temp = re.sub('(\d+)([a-zA-Z])', r'\1*\2', stringybingy)
        # x69 -> 69*x, x**2y**2 -> x**2*y**2
        temp = re.sub('([a-zA-Z])(\d+)', r'\2*\1', temp)
        # 4xyz+2yz -> 4x*y*z+2y*z
        for lol in re.findall('[a-zA-Z]{2,}', temp):
            temp = temp.replace(lol,'*'.join(lol))
        # x^2 -> x**2
        temp = re.sub('\^','**',temp)
        temp = re.sub('{(\d+|\d+.\d+)}',r'\1',temp)
        return temp

    def __repr__(self):
        return self.strfunc

    def __call__(self, X):
        return self.f(*X.reshape(-1, *X.shape).T)

    def _get_domain(self):
        '''Think of this as a multidimensional np.linspace'''
        stds = self.X.std(axis=0)/4
        mins = self.X.min(axis=0)
        maxs = self.X.max(axis=0)
        D = np.meshgrid(*[np.linspace(m-s,x+s,20) for m, x, s in
                         zip(mins, maxs, stds)])
        points = np.array([d.ravel() for d in D]).T
        return points, D

    def _show2d(self, figsize: tuple, show: bool=True):
        '''
        Case for f(x)
        '''
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(self.X, self.y, s=16, label='Datapoint')
        ax.set_title(rf'${self.strfunc}$')

        points, _ = self._get_domain()
        ax.plot(points.ravel(), self(points), c='red', label='True')
        ax.grid()

        plt.tight_layout()

        # title = figdict[self.strfunc]
        # titley = title.split('.')[0]
        # ax.set_title(f'${titley}={self.strfunc}$', fontsize=16)
        # plt.savefig(figdict[self.strfunc]+'_2d.pdf')
        if show: 
            plt.show()
        else:
            return fig, ax, points

    def _show3d(self, figsize: tuple, show: bool=True):
        '''
        Case for f(x,y)
        '''
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(*self.X.T, self.y, label='Datapoint')
        ax.set_title(rf'${self.strfunc}$')

        points, D = self._get_domain()
        ax.plot_wireframe(D[0], D[1],
                          self(points).reshape(D[0].shape),
                          color='red', alpha=0.2, label='True')

        plt.setp(ax, xlabel=self.vars[0], ylabel=self.vars[1], zlabel='z')
        plt.tight_layout()

        # title = figdict[self.strfunc]
        # titley = title.split('.')[0]
        # ax.set_title(f'${titley}={self.strfunc}$', fontsize=14)
        # plt.savefig(figdict[self.strfunc]+'_3d.pdf')
        if show: 
            plt.show()
        else: 
            return fig, ax, D

    def show(self, figsize=(5,3)):
        '''
        Visulize generated data and polynomial
        '''
        if self.n_vars == 1: self._show2d(figsize)
        elif self.n_vars == 2: self._show3d(figsize)
        else: raise RuntimeError('Too many variables to visualize')

    def generate_data(self, n: int=128, std: float=10,
                      sampler: Callable=None, show: bool=False,
                      figsize: tuple=(7,5), n_noise: int=0, **xkwargs):
        '''
        Generate dummy data for regression using the function

        Input
        -----------
        n: number of datapoints
        sampler: sampling function from np.random, if None, then standard
                 normal is selected
        show: To visualize the function with the generated data or not
        figsize: size of figure
        n_noise: whether to include noise features
        kwargs gets passed to sampler function

        Returns
        ----------
        X, y
        '''    

        if sampler is None:
            # Default is np.random.normal
            sampler = np.random.normal

        # Only generate data if never generated before
        if self.X is None:
            self.X = sampler(size=(n, self.n_vars), **xkwargs)
            self.y = self.f(*self.X.T) + np.random.randn(n)*std
            X = self.X

        if n_noise > 0:
            noise = np.random.normal(
                loc=np.random.uniform(low=-10, high=10, size=1),
                scale=np.random.uniform(low=0.1, high=10, size=1),
                size=(n, n_noise)
            )
            self.X_noised = np.column_stack([self.X, noise])
            X = self.X_noised
        else:
            self.X_noised = None
            
        if show: self.show()
        return X, self.y

    def get_degree(self):
        '''Get order of highest order factor'''
        factors = re.findall('[a-zA-Z]\*\*(\d+)', self.python_strfunc)
        if len(factors) == 0:
            # No higher orders
            return 1
        if len(factors) > 0:
            # There exists higher order factors
            return int(max(factors))
            
    def get_weights(self, weight_dict: dict):
        '''
        Given a dictionary of factors and weights, i.e factors as keys
        and weights (coefficients) as values, returns an array weights
        that is compitable with Sklearn's polynomial features. 
        '''
        PolyFeats = PolynomialFeatures(degree=self.get_degree())\
               .fit(np.zeros((2,self.n_vars)))
        features = PolyFeats.get_feature_names(self.vars)
        features = [feat.replace(' ','') for feat in features]
        weights = [weight_dict.setdefault(feat, 0) for feat in features]
        return np.array(weights)

    def _w_hat2d(self, figsize: tuple, response: np.ndarray):
        fig, ax, points = self._show2d(figsize, show=False)
        ax.plot(points.ravel(), response, color='navy', label='Estimate')

    def _w_hat3d(self, figsize: tuple, response: np.ndarray):
        fig, ax, D = self._show3d(figsize, show=False)
        ax.plot_wireframe(D[0], D[1],
                          response.reshape(D[0].shape),
                          color='navy', alpha=0.2, label='Estimate')
    
    def plot_w_hat(self, model: LinearRegression, figsize: tuple=(6,3),
                   filename: str = None):
        '''
        Plots estimated polynomials against the data and the actual polynomial
        '''
        print(self.strfunc)
        w_hat = model.w_hat
        points, D = self._get_domain()
        PolyTransformer = PolynomialFeatures(degree=self.get_degree())
        points = PolyTransformer.fit_transform(points)
        
        if self.X_noised is not None:
            d = self.X_noised.shape[0]
            names = np.arange(2,d).astype('str')
            A = PolyTransformer.get_feature_names(names)
            B = model.PolyTransformer.get_feature_names(names)
            
            idxs = np.argmax(np.repeat(np.array([A]).T, len(B), axis=1) == \
            np.repeat([B], len(A), axis=0), axis=1)
            w = w_hat[idxs]
        else:
            w = w_hat[:points.shape[1]]

        response = w@points.T

        if self.n_vars == 1:
            self._w_hat2d(figsize, response)
        elif self.n_vars == 2:
            self._w_hat3d(figsize, response)
        else:
            raise ValueError('Too many dimensions')
        plt.legend()
        if filename is not None: plt.savefig(filename)
        # plt.show()
        plt.close()

if __name__ == '__main__':
    import re 
    import pandas as pd 
    from pprint import pprint 

    # Functions should be written in simple form, 
    # i.e write 2x + 2x as 4x, 40+40 as 80, 3xy+2xy as 5xy etc...
    polynomials=[
        '2x+69',
        '4x^3+2x^2-420',
        '-4x+20y+69',
        '69x+69y-420',
        'x^2-2y^2',
        'xy-x^2+y^2-420',
        '-x^3+42x^2-20x-y^3+42y^2-20y+69',
        '2x^3-y^3-3xy^2+3x^2y+x^3-3yx+69',
    ]
    
    # Used when saving plots to dicts
    figdict = {eq:f'f_{i}' for i, eq in enumerate(polynomials, start=1)}

    normal = np.random.normal
    expo = np.random.exponential
    uniform = np.random.uniform

    kwarglist=[
        dict(show=False, std=9, sampler=expo, scale=1/0.1),
        dict(show=False, std=12, sampler=normal, loc=-1.69, scale=1.2),
        dict(show=False, std=22, sampler=expo, scale=1/0.2),
        dict(show=False, std=420, sampler=normal, loc=69, scale=4),
        dict(show=False, std=69, sampler=uniform, low=-16, high=16),
        dict(show=False, std=69, sampler=normal, loc=0, scale=8),
        dict(show=False, std=420, sampler=uniform, low=-2, high=36),
        dict(show=False, std=69, sampler=uniform, low=-6, high=6)
    ]

    def simulation():
        '''
        Run this function to execute simulation task
        '''
        get_polyfunc_datas = lambda fs, n_noise, kwarglist: \
            [f.generate_data(n_noise=n_noise, **kwargs) 
             for f, kwargs in zip(fs, kwarglist)]

        polyfuncs=[PolyFunc(poly) for poly in polynomials]

        sim_data1=get_polyfunc_datas(polyfuncs, 0, kwarglist)
        sim_data2=get_polyfunc_datas(polyfuncs, 1, kwarglist)
        sim_data3=get_polyfunc_datas(polyfuncs, 2, kwarglist)
        sim_data4=get_polyfunc_datas(polyfuncs, 3, kwarglist)
        
        dicto=dict(
            noise_features=[],
            function=[f'$f_{i}$' for i in range(1,len(polynomials)+1)]*4,
            degree=[],
            ARD_MAE=[],
            Regular_MAE=[],
            Regular_over_ARD=[]
        )   

        ards = []
        regs = []
        for i, fs, sim_datas in zip(range(4), [polyfuncs]*4, 
                    [sim_data1, sim_data2, sim_data3, sim_data4]):
            temp_ards = []
            temp_regs = []
            for f, data in zip(fs, sim_datas):
                deg = int(f.get_degree())
                ard = LinearRegressionARD(
                    n_gibbs=2000, polydegree=deg).fit(*data)
                regular = LinearRegression(
                    n_gibbs=2000, polydegree=deg).fit(*data)
        
                temp_ards.append(ard)
                temp_ards.append(regular)

                # Render estimated polynomials
                title = figdict[f.strfunc]
                filename = 'images/'+\
                    title + f'_noise{i}_estplot_ard.pdf'
                f.plot_w_hat(ard, filename=filename)
                filename = 'images/'+\
                    title + f'_noise{i}_estplot_regular.pdf'
                f.plot_w_hat(regular, filename=filename)

                # Rebder traceplots
                title = figdict[f.strfunc]
                filename = 'images/'+\
                    title + f'_noise{i}_tplot_ard.pdf'
                ard.plot_result(axes=[0,1], figsize=(5,6), 
                                figtitle=f'${title}$', filename=filename)
                filename = 'images/'+\
                    title + f'_noise{i}_tplot_reg.pdf'
                regular.plot_result(axes=[0,1], figsize=(5,6), 
                                figtitle=f'${title}$', filename=filename)
    
                ard_mae = ard.get_mae()
                regular_mae = regular.get_mae()

                dicto['noise_features'].append(i)
                dicto['degree'].append(deg)
                dicto['ARD_MAE'].append(ard_mae)
                dicto['Regular_MAE'].append(regular_mae)
                dicto['Regular_over_ARD'].append(regular_mae/ard_mae)

            ards.append(temp_ards)
            regs.append(temp_regs)

        df = pd.DataFrame(dicto).set_index(['noise_features','function'])
        latex = df.to_latex(index=True, float_format=lambda x: f'{x:.2f}')\
                    .replace(r'\$','$')\
                    .replace('f\\_','f_')\
                    .replace(r'\textasciicircum ','^')

        print(latex)
        df.to_csv('simulation_results_mae.csv')

    # simulation()

    def real():
        '''
        Run this function to do task on real data
        '''
        from sklearn.ensemble import RandomForestRegressor
        df = pd.read_csv('Lung_cancer_small.csv')
        X = df.values[:,1:]
        y = df.values[:,0]
        
        # To store feature importance data
        feat_dict = {}
        feat_dict['Feature'] = []
        feat_dict['coefficient'] = []
        feat_dict['With ARD'] = ['ARD']*5 + ['without ARD']*5

        ard = LinearRegressionARD(polydegree=1, n_gibbs=2000)
        ard.fit(X, y)
        fig, _ = ard.plot_w_traces(axes=np.arange(15), n_cols=5, ticks=True, 
                          show=False)
        fig.suptitle('ARD')
        # plt.savefig('images/real_w_traces_ard.pdf')
        plt.close()

        fig, _ = ard.plot_alpha_traces(axes=np.arange(15), n_cols=5, 
                                       ticks=True, show=False)
        fig.suptitle('ARD alphas')
        # plt.savefig('images/real_alpha_traces_ard.pdf')
        plt.close()

        print(ard.get_mae())
        # Ignore intercept and then argsort wrt to absolute values
        argsort = abs(ard.w_hat[1:]).argsort()
        # Get first five features with highest coefficients
        feat_dict['Feature'].extend(list(df.columns[1:][argsort[::-1]][:5]))
        feat_dict['coefficient'].extend(list(ard.w_hat[1:][argsort[::-1]][:5]))

        regular = LinearRegression(polydegree=1, n_gibbs=2000)
        regular.fit(X, y)
        fig, _ = regular.plot_w_traces(axes=np.arange(15), n_cols=5, 
                                       ticks=True, show=False)
        fig.suptitle('Without ARD')
        # plt.savefig('images/real_w_traces_regular.pdf')
        # plt.close()
        print(regular.get_mae())
        
        # Ignore intercept and then argsort wrt to absolute values
        argsort = abs(regular.w_hat[1:]).argsort()
        # Get first five features with highest coefficients
        feat_dict['Feature'].extend(list(df.columns[1:][argsort[::-1]][:5]))
        feat_dict['coefficient'].extend(list(ard.w_hat[1:][argsort[::-1]][:5]))

        df_features = pd.DataFrame(feat_dict).set_index(['With ARD', 'Feature'])
        latex = df_features.to_latex(index=True, float_format=lambda x: f'{x:.2f}')\
                    .replace(r'\$','$')\
                    .replace('f\\_','f_')\
                    .replace(r'\textasciicircum ','^')
        print(latex)

    real()
