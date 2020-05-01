/*
Matrix factorization using normal distributions
Assuming we use column vectors:

Approximates two matrices U,V s.t X \approx U^TV.
where U, V  have less dimensions than X

Note that U^T in the code is denoted as U even
though it is technically U^T
*/

data {
    int<lower=1> n_components; // Dimension of embeddings
    int<lower=0> n; // rows in data matrix
    int<lower=0> df[n,3]; // data matrix (is df.values)
                          // rows should be [row_idx, col_idx, rating]

    int<lower=0> p; // Dense matrix representation
    int<lower=0> q; // dimensions, i.e shape(D) = p x q

    real mu_u; // Mean for U elements
    real mu_v; // Mean for V elements

    // Prior parameters for alpha ~ gamma(a_alpha, b_alpha)
    real<lower=0> a_alpha;
    real<lower=0> b_alpha;
    
    // Prior parameters for beta ~ gamma(a_beta, b_beta)
    real<lower=0> a_beta;
    real<lower=0> b_beta;
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
    matrix<lower=0>[p, n_components] U;
    matrix<lower=0>[q, n_components] VT;
    vector<lower=0>[n_components] alpha;
    real<lower=0> beta;
}

model {
    matrix[p,q] X_hat;
    int row_idx;
    int col_idx;
    int rating;
    int R[3];

    to_vector(U) ~ normal(mu_u, 1);
    to_vector(VT) ~ normal(mu_v, 1);
    alpha ~ gamma(a_alpha, b_alpha);
    beta ~ gamma(a_beta, b_beta);

    X_hat = diag_post_multiply(U, alpha)*diag_post_multiply(VT, alpha)';

    for (i in 1:n) {
        R = X[i];
    
        row_idx = R[1];
        col_idx = R[2];
        rating = R[3];

        rating ~ normal(X_hat[row_idx, col_idx], beta);
    }
}