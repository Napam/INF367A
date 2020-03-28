import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt 
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import norm
from common import texmatrix
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model.base import BaseEstimator
import sklearn.metrics as metrics

def get_data():
    pd.set_option('display.expand_frame_repr', False)
    df = pd.read_csv('new_york_bicycles3.csv')

    X = df.values[:,1:]
    y = df.values[:,0]
    return X, y 

def poly_basis(degree: int=2):
    def kernelized(X):
        return PolynomialFeatures(degree, include_bias=True).fit_transform(X)
    return kernelized

def get_thetas(Xs, beta):
    '''
    Calculate thetas (using conjugate prior)
    '''
    thetas = []
    for X_ in Xs:
        d = X_.shape[1]

        A = alpha*np.eye(d) + beta * X_.T@X_
        S = np.linalg.inv(A)
        theta = m = S@(beta * (X_ * y_)).sum(axis=0) # This is theta hat 
        thetas.append(theta)
    return thetas

def get_loglikelihoods(Xs, thetas, beta):
    '''
    Bayesian way 
    '''
    lhoods = []
    for theta, X_ in zip(thetas, Xs):
        preds = theta@X_.T
        logL = np.log(norm.pdf(preds-y, 0, np.sqrt(1/beta))+1e-12).sum()
        lhoods.append(logL)
    return lhoods

class LinReg(BaseEstimator):
    '''
    Sklearn compitable Linear Regression that 
    fits parameters using bayesian way (not used)
    '''
    def __init__(self, beta=1):
        self.beta = beta
    
    def fit(self, X, y, *args, **kwargs):
        d = X.shape[1]

        A = alpha*np.eye(d) + self.beta * X.T@X
        S = np.linalg.inv(A)
        theta = m = S@(self.beta * (X * y.reshape(-1,1))).sum(axis=0) # This is theta hat 
        self.theta = theta

    def predict(self, X):
        return self.theta@X.T

if __name__ == '__main__':
    X, y = get_data()
    y_ = y.reshape(-1,1)

    alpha=1
    n = X.shape[0]
    betas = np.linspace(0.1,6,5)

    phis = [poly_basis(i) for i in range(1, 9)]

    degrees= []
    betas_ = []
    bics = []
    aics = []
    lhoods = []
    cv_vals = []
    cv_trains = []
    for beta in betas:
        Xs = [phi(X) for phi in phis]
        thetas = get_thetas(Xs, beta)
        loglikelihoods = get_loglikelihoods(Xs, thetas, beta=beta)

        def get_bics(Xs, thetas, lhoods):
            BICs = []
            for X_, theta, logL in zip(Xs, thetas, lhoods):
                d = X_.shape[1]
                other_term = d/2 * np.log(n)
                BICs.append(logL - other_term)
            return BICs

        def get_aics(Xs, thetas, lhoods):
            AICs = []
            for X_, theta, logL in zip(Xs, thetas, lhoods):
                AICs.append(-2/n * logL + 2*X_.shape[1]/n)
            return AICs

        def get_cvs(Xs, thetas, lhoods):
            mean_val_scores = []
            mean_train_scores = []
            for X_, theta in zip(Xs,thetas):
                gsc = GridSearchCV(LinearRegression(fit_intercept=False), param_grid={}, scoring='neg_mean_absolute_error', cv=5, iid=True, return_train_score=True)
                gsc.fit(X_, y)       
                mean_val_scores.append(gsc.cv_results_['mean_test_score'][0])
                mean_train_scores.append(gsc.cv_results_['mean_train_score'][0])
            return mean_val_scores, mean_train_scores

        degrees.extend(np.arange(1,9))
        betas_.extend(np.full(8, beta))
        bics.extend(get_bics(Xs, thetas, loglikelihoods))
        aics.extend(get_aics(Xs, thetas, loglikelihoods))
        lhoods.extend(loglikelihoods)
        cv_vals_, cv_trains_ = get_cvs(Xs, thetas, loglikelihoods)
        cv_vals.extend(cv_vals_)
        cv_trains.extend(cv_trains_)

    print(len(bics))
    print(len(aics))
    print(len(cv_vals))
    print(len(cv_trains))
    
    df = pd.DataFrame(dict(
        degree=degrees,
        beta=betas_,
        LogLhoods=lhoods,
        BIC=bics,
        AIC=aics,
        CV_val=cv_vals,
        CV_train=cv_trains
    ))

    pd.options.display.float_format = '{:.2f}'.format
    # print(df.to_latex(index=False))
    print(df.set_index(['beta','degree']).to_latex())
    

