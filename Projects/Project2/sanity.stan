/*
Matrix factorization using normal distributions

Approximates two matrices U,V s.t X \approx U^TV.
where U, V  have less dimensions than X

Note that U^T in the code is denoted as U even
though it is techinically U^T
*/

data {
    int<lower=1> n_components; // Dimension of embeddings
    int<lower=0> n; // rows in data matrix
    int<lower=0> m; // columns in data matrix
    int X[n,m]; // data matrix

    int<lower=0> p; // Dense matrix representation
    int<lower=0> q; // dimensions

    real mu_u; // Prior mean of elements in U matrix
    real<lower=0> sigma_u; // Prior std of elements in U matrix

    real mu_v; // Prior mean of elements in V matrix
    real<lower=0> sigma_v; // Prior std of elemens in V matrix

    real<lower=0> sigma_x; // rating ~ N(U*V, sigma_x) 
}

parameters {
    matrix[p, n_components] U;
    matrix[n_components, q] V;
}

model {
    matrix[p,q] X_hat;
    int row_idx;
    int col_idx;
    int rating;
    int R[3];

    to_vector(U) ~ normal(mu_u, sigma_u);
    to_vector(V) ~ normal(mu_v, sigma_v);

    X_hat = U*V;

    for (i in 1:n) {
        R = X[i];
    
        row_idx = R[1];
        col_idx = R[2];
        rating = R[3];

        rating ~ normal(X_hat[row_idx, col_idx], sigma_x);
    }
}