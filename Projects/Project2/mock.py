import numpy as np
seed=42069
np.random.seed(seed)
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
import sys 
from scipy import sparse 
from matplotlib import pyplot as plt 


np.set_printoptions(precision=2, suppress=True)

A, y = make_blobs(n_samples=12, n_features=6)

import re 

txt = '''
/*
Matrix factorization using normal distributions
Assuming we use column vectors:

Approximates two matrices U,V s.t X \approx U^TV.
where U, V  have less dimensions than X

Note that U^T in the code is denoted as U even
though it is technically U^T
*/

/**/

data {
    int<lower=1> n_components; // Dimension of embeddings
    int<lower=0> n; // rows in data matrix
    int<lower=0> df[n,3]; // data matrix (is df.values)
                          // rows should be [row_idx, col_idx, rating]

    int<lower=0> p; // Dense matrix representation
    int<lower=0> q; // dimensions, i.e shape(D) = p x q

    // Prior parameters for U ~ gamma(a_u, b_u)
    real<lower=0> a_u; 
    real<lower=0> b_u; 

    // Prior parameters for V ~ gamma(a_v, b_v)
    real<lower=0> a_v; 
    real<lower=0> b_v; 

    // Prior parameters for beta ~ gamma(a_beta, b_beta)
    real<lower=0> a_beta; 
    real<lower=0> a_beta; 
}

transformed data {
    // Increment index values with 1 because
    // Stan is lame 
    int X[n,3] = df;

    for (i in 1:n) {
        X[i,1] += 1;
        X[i,2] += 1;
    }
}

parameters {
    matrix[p, n_components] U;
    matrix[n_components, q] V;
    real<lower=0> beta;
}

model {
    matrix[p,q] X_hat;
    int row_idx;
    int col_idx;
    int rating;
    int R[3];

    to_vector(U) ~ gamma(a_u, b_u);
    to_vector(V) ~ gamma(a_v, b_v);
    beta ~ gamma(a_beta, b_beta)

    X_hat = U*V;

    for (i in 1:n) {
        R = X[i];
    
        row_idx = R[1];
        col_idx = R[2];
        rating = R[3];

        rating ~ normal(X_hat[row_idx, col_idx], beta);
    }
}
'''

txt = re.sub('(\/\*(.|\n)*?\*\/)', '', txt)
txt = re.sub('(//.+)', '', txt)

txt = re.sub('([ \t]{2,})', ' ', txt)
txt = re.sub('(\s+\n)', '\n', txt)
txt = re.sub('(\n{2,})', '\n', txt)

print(txt)



