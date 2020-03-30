import numpy as np
# import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import gamma, norm
from scipy.stats import multivariate_normal as mvn
from typing import Callable, Union
from collections import defaultdict
np.random.seed(42069)
from tqdm import tqdm

_eps = 1e-32

class LinearRegression:
    '''
    Linear regression using Gibbs Sampling without ARD
    '''
    def __init__(self, init_ws: np.ndarray=None, init_beta: float=None,
                 polydegree: int=1, alpha: float=1, a: float=0.01, b: float=0.01,
                 n_gibbs: int=42069):
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
        m = self.betas[i]*S@(self.X*self.y.reshape(-1,1)).sum(axis=0)
        return mvn.rvs(mean=m, cov=S)

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
        mvn.logpdf(self.ws[i], mean=origo, cov=self.D_cov).clip(_eps).sum()

        log_beta =\
        gamma.logpdf(self.betas[i], a=self.a, scale=1/self.b).clip(_eps).sum()

        log_data =\
        norm.logpdf(self.y, loc=self.ws[i]@self.X.T, 
                    scale=np.sqrt(1/self.betas[i])).clip(_eps).sum()
            
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
            self.init_ws = np.random.uniform(0.1, 2, self.dim)
        if self.init_beta is None:
            self.init_beta = np.random.uniform(0.1, 2, 1)

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
            hist, edges = np.histogram(w[self.n_gibbs//4:], bins='scott')
            self.w_hat[i]=edges[hist.argmax()+1]
        return self.w_hat

    def _determine_beta(self):
        '''
        Determines beta_hat by taking argmax of histogram
        '''
        hist, edges = np.histogram(self.betas[self.n_gibbs//4:], bins='scott')
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
            self.plot_dict[rf'$w_{i}$'] = self.ws[skip:,i]

        self.plot_kwargs = {'linewidth':0.69}
        # Histogram bin count calculated using Scott's rule
        self.hist_kwargs = {'bins':'auto', 'edgecolor':'k', 'linewidth':0.69}

    def plot_result(self, figsize:tuple = (9,9), axes: list=[0,1],
                    skip: int=10, show: bool=True) -> None:
        '''
        Visualize results.

        The plots discards the datapoints from the first
        two iterations as they really mess up the plots. That is because
        they are outliers with vastly different values from the rest, but
        pyplot will still scale the plots such that they are included.
        '''
        self._init_plot(figsize, axes, skip)
        for i, title, data in zip(range(len(self.axes)),
                                 self.plot_dict.keys(),
                                 self.plot_dict.values()):
            self.axes_traces[i].set_title(title+' trace')

            self.axes_traces[i].plot(data[self.n_gibbs//4:], **self.plot_kwargs)

            self.axes_hists[i].set_title(title+' hist')
            self.axes_hists[i].hist(data[self.n_gibbs//4:], **self.hist_kwargs)

        self.fig.tight_layout()
        if show: plt.show()

class LinearRegressionARD(LinearRegression):
    '''
    Linear regression with automatic relevance determination.
    Fitted using Gibbs sampling.
    '''
    def __init__(self, init_ws: np.ndarray=None, init_beta: float=None,
                 init_alphas:np.ndarray=None, polydegree: int=1, a: float=0.01,
                 b: float=0.01, c: float=0.01, d: float=0.01, n_gibbs: int=42069):
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
        norm.logpdf(self.ws[i], loc=0, scale=np.sqrt(1/self.alphas[i]))\
            .clip(_eps).sum()

        log_alphas =\
        gamma.logpdf(self.alphas[i], a=self.c, scale=1/self.d).clip(_eps).sum()

        log_betas =\
        gamma.logpdf(self.betas[i], a=self.a, scale=1/self.b).clip(_eps).sum()

        log_data =\
        norm.logpdf(self.y, loc=self.ws[i]@self.X.T, 
                    scale=np.sqrt(1/self.betas[i])).clip(_eps).sum()
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
        if self.init_ws is None:
            self.init_ws = np.random.uniform(0.1, 2, self.dim)
        if self.init_beta is None:
            self.init_beta = np.random.uniform(0.1, 2, 1)
        if self.init_alphas is None:
            self.init_alphas = np.random.uniform(0.1, 2, self.dim)

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
        self.alpha_hat = np.empty(self.dim)
        for i, w in enumerate(self.ws.T):
            hist, edges = np.histogram(w[self.n_gibbs//4], bins='scott')
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
        self.hist_kwargs = {'bins':'scott', 'edgecolor':'k', 'linewidth':0.69}

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
        stds = self.X.std(0)/4
        mins = self.X.min(0)
        maxs = self.X.max(0)
        D = np.meshgrid(*[np.linspace(m-s,x+s,20) for m, x, s in
                         zip(mins, maxs, stds)])
        points = np.array([d.ravel() for d in D]).T
        return points, D

    def _show2d(self, figsize: tuple=(4,3)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(self.X, self.y, s=16)
        ax.set_title(rf'${self.strfunc}$')

        points, _ = self._get_domain()
        ax.plot(points.ravel(), self(points), c='red')

        plt.tight_layout()
        plt.show()

    def _show3d(self, figsize=(4,3), show: bool=True):
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(*self.X.T, self.y)
        ax.set_title(rf'${self.strfunc}$')

        points, D = self._get_domain()
        ax.plot_wireframe(D[0], D[1],
                          self(points).reshape(D[0].shape),
                          color='red', alpha=0.2)

        plt.setp(ax, xlabel=self.vars[0], ylabel=self.vars[1], zlabel='z')
        plt.tight_layout()

        # title = figdict[self.strfunc]
        # titley = title.split('.')[0]
        # ax.set_title(f'${titley}={self.strfunc}$')
        # plt.savefig(figdict[self.strfunc])
        if show: 
            plt.show()
        else: 
            return fig, ax

    def show(self, figsize=(5,4)):
        if self.n_vars == 1: self._show2d(figsize)
        elif self.n_vars == 2: self._show3d(figsize)
        else: raise RuntimeError('Too many variables to visualize')

    def generate_data(self, n: int=128, std: float=10,
                      x_sampler: str='uniform', show: bool=False,
                      figsize: tuple=(7,5), **xkwargs):
        '''
        Generate dummy data for regression using the function
        '''
        if x_sampler == 'normal':
            self.X = np.random.randn(n, self.n_vars)\
                *xkwargs['sigma']+xkwargs['mean']
        if x_sampler == 'uniform':
            self.X = np.random.uniform(size=(n, self.n_vars), **xkwargs)
        if x_sampler == 'expo':
            self.X = np.random.exponential(size=(n, self.n_vars), **xkwargs)

        self.y = self.f(*self.X.T) + np.random.randn(n)*std

        if show: self.show()

        return self.X, self.y

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
        # for feature, weight in zip(features, weights):
            # print(feature,':',weight)
        # print()

    def plot_w_hat(self, model: LinearRegression):
        points, D = self._get_domain()
        w_hat = model.w_hat

        points = PolynomialFeatures(degree=self.get_degree()).fit_transform(points)

        w = w_hat[:points.shape[1]]
        z = w@points.T

        fig, ax = self._show3d(show=False)
        
        ax.plot_wireframe(D[0], D[1],
                          z.reshape(D[0].shape),
                          color='navy', alpha=0.2)
        plt.show()

def gibbs_MAE(model: LinearRegression, n: int=64):
    '''
    Choose n different sets of model parameters 
    and then calculate the mean of the absolute predictive errors 
    of each set if model parameters. 
    '''
    ws = model.ws[model.n_gibbs//4:]
    ws = ws[np.random.choice(np.arange(model.n_gibbs//4), size=n, 
                             replace=False)]
    y_hats = model.X@ws.T
    mae_wrt_ws = np.mean(abs((y_hats - model.y.reshape(-1,1))), axis=0)
    return mae_wrt_ws

if __name__ == '__main__':
    import re 
    import pandas as pd 
    from pprint import pprint 

    def feat_names(gibby):
        names=gibby.PolyTransformer\
                   .get_feature_names(['x','y','z','u','v','t'])
        vals=gibby.w_hat
        for name, val in zip(names,vals):
            print(name,':',val)
        
    def poly_from_dicts(dicts):
        gen_polys = []
        for stuff in weight_dicts:
            polystring = ''
            for key, val in stuff.items():
                if val >= 0:
                    if val == 1:
                        polystring += '+'+key
                    else:
                        polystring += '+'+str(val)+key
                else: 
                    if val == -1:
                        polystring += '-'+key
                    else:
                        polystring += str(val)+key
                
            if polystring[0] == '+':
                polystring = polystring.strip('+')
            polystring = polystring.rstrip('1')
            gen_polys.append(polystring)
        return gen_polys

    # Functions should be written in simple form, 
    # i.e write 2x + 2x as 4x, 40+40 as 80, 3xy+2xy as 5xy etc...
    polynomials=[
        '-4x+20y+69',
        '69x+69y-420',
        'x^2-2y^2',
        'xy-x^2+y^2-420',
        '-x^3+42x^2-20x-y^3+42y^2-20y+69',
        '2x^3-y^3-3xy^2+3x^2y+x^3-3yx+69',
    ]

    # I kind of shot myself in the foot by making the PolyFunc class
    # It is really difficult to calculate parameter mse using it.
    # This is not optimal, but I need a way to calculate parameter mse. 
    # The dictionary contains factors as keys, and coefficients as values
    weight_dicts = [
        {'x':-4, 'y':20 ,'1':69,},
        {'x':69, 'y':69 ,'1':-420,},
        {'x^2':1, 'y^2':-2 ,'1':0,},
        {'xy':1, 'x^2':-1, 'y^2':1, '1':-420,},
        {'x^3':-1, 'x^2':42, 'x':-20, 'y^3':-1, 'y^2':42, 'y':-20 ,'1':69,},
        {'x^3':2, 'y^3':-1, 'xy^2':-3, 'x^2y':3, 'x^3':1, 'yx':-3 ,'1':69,},
    ]
        
    figdict = {eq:f'f_{i}.pdf' for i, eq in enumerate(polynomials, start=1)}

    kwarglist=[
        dict(show=False, std=22, x_sampler='expo', scale=1/0.2),
        dict(show=False, std=420, x_sampler='normal', mean=69, sigma=4),
        dict(show=False, std=69, x_sampler='uniform', low=-16, high=16),
        dict(show=False, std=22, x_sampler='normal', mean=0, sigma=8),
        dict(show=False, std=420, x_sampler='expo', scale=1/0.15),
        dict(show=False, std=69, x_sampler='uniform', low=-6, high=6)
    ]

    def simulation2():
        get_polyfunc_datas = lambda fs, kwarglist: \
            [f.generate_data(**kwargs) for f, kwargs in zip(fs, kwarglist)]

        fs1=[PolyFunc(poly) for poly in polynomials]
        sim_data1=get_polyfunc_datas(fs1, kwarglist)

        # PolyFunc objects beneath will generate noise features
        fs2=[PolyFunc(poly+'+0z') for poly in polynomials]
        sim_data2=get_polyfunc_datas(fs2, kwarglist)

        fs3=[PolyFunc(poly+'+0z+0u') for poly in polynomials]
        sim_data3=get_polyfunc_datas(fs3, kwarglist)
        
        fs4=[PolyFunc(poly+'+0z+0u+0v') for poly in polynomials]
        sim_data4=get_polyfunc_datas(fs4, kwarglist)
        
        dicto=dict(
            noise_features=[],
            function=[f'$f_{i}$' for i in range(1,len(polynomials)+1)]*4,
            degree=[],
            ARD_MAE=[],
            Regular_MAE=[],
            Regular_over_ARD=[]
        )   

        mse = lambda w_true, w_hat: np.mean((w_true-w_hat)**2)

        ards = []
        regs = []
        for i, fs, sim_datas in zip(range(4), [fs1, fs2, fs3, fs4], 
                    [sim_data1, sim_data2, sim_data3, sim_data4]):
            temp_ards = []
            temp_regs = []
            for f, data, weight_dict in zip(fs, sim_datas, weight_dicts):
            # for f, data, weight_dict in tqdm(zip(fs, sim_datas, weight_dicts)):
                deg = int(f.get_degree())
                JohnWick = LinearRegressionARD(
                    n_gibbs=512, polydegree=deg).fit(*data)
                JohnCena = LinearRegression(
                    n_gibbs=512, polydegree=deg).fit(*data)
        
                temp_ards.append(JohnWick)
                temp_ards.append(JohnCena)
    
                ard_mae = gibbs_MAE(JohnWick).mean()
                regular_mae = gibbs_MAE(JohnCena).mean()

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
        df.to_csv('simulation_results_mse.csv')

    simulation2()

    def simulation_old():
        fs1=[PolyFunc(poly) for poly in polynomials]
        sim_data1=[f.generate_data(**kwargs) for f, kwargs in zip(fs1, kwarglist)]

        # PolyFunc objects beneath will generate noise features
        fs2=[PolyFunc(poly+'+0z') for poly in polynomials]
        sim_data2=[f.generate_data(**kwargs) for f, kwargs in zip(fs2, kwarglist)]

        fs3=[PolyFunc(poly+'+0z+0u') for poly in polynomials]
        sim_data3=[f.generate_data(**kwargs) for f, kwargs in zip(fs3, kwarglist)]
        
        fs4=[PolyFunc(poly+'+0z+0u+0v') for poly in polynomials]
        sim_data4=[f.generate_data(**kwargs) for f, kwargs in zip(fs4, kwarglist)]
        
        fs5=[PolyFunc(poly+'+0z+0u+0v+0t') for poly in polynomials]
        sim_data5=[f.generate_data(**kwargs) for f, kwargs in zip(fs5, kwarglist)]


        samplers = ['Unif(-2,20)', 
                    'Exp(5)', 
                    'N(69, 4)', 
                    'N(0,8)', 
                    'Exp(5)', 
                    'Unif(-6,6)' 
                    ]*5

        dicto=dict(
            noise_features=[],
            function=[f'$f_{i}$' for i in range(1,len(polynomials)+1)]*5,
            degrees=[],
            ARD_BICS=[],
            Regular_BICS=[],
            ARD_AICS=[],
            Regular_AICS=[],
        )   

        from tqdm.auto import tqdm

        ards = []
        regs = []
        for i, fs, sim_datas in zip(range(5), [fs1, fs2, fs3, fs4, fs5], 
                        [sim_data1, sim_data2, sim_data3, sim_data4, sim_data5]):
            temp_ards = []
            temp_regs = []
            for f, data in tqdm(zip(fs, sim_datas)):
                deg = int(f.get_degree())
                JohnWick = LinearRegressionARD(
                    n_gibbs=2048*2, polydegree=deg).fit(*data)
                Gogoplex = LinearRegression(
                    n_gibbs=2048*2, polydegree=deg).fit(*data)

                temp_ards.append(JohnWick)
                temp_regs.append(Gogoplex)

                dicto['noise_features'].append(i)
                dicto['degrees'].append(deg)
                dicto['ARD_BICS'].append(JohnWick.get_bic())
                dicto['Regular_BICS'].append(Gogoplex.get_bic())
                dicto['ARD_AICS'].append(JohnWick.get_aic())
                dicto['Regular_AICS'].append(Gogoplex.get_aic())

            ards.append(temp_ards)
            regs.append(temp_regs)

        df = pd.DataFrame(dicto).set_index(['noise_features','function'])
        latex = df.to_latex(index=True, float_format=lambda x: f'{x:.2f}')\
                    .replace(r'\$','$')\
                    .replace('f\\_','f_')\
                    .replace(r'\textasciicircum ','^')

        print(latex)
        df.to_csv('simulation_results.csv')

    def real():
        df = pd.read_csv('Lung_cancer_small.csv')
        X = df.values[:,1:]
        y = df.values[:,0]

        boi = LinearRegressionARD(polydegree=1, n_gibbs=3000)
        boi.fit(X, y)
        boi.plot_result(axes=[])
        return boi 
        # boi.plot_result(axes=[])

    # boi = real()