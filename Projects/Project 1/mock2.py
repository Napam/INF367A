from scipy.stats import gamma, norm
import numpy as np 
from matplotlib import pyplot as plt 
from sklearn.preprocessing import PolynomialFeatures

A = norm.logpdf([1,2,3,4], loc=0, scale=[1,2,3,4])
print(A)