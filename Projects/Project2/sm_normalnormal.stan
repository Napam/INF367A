/*
Matrix factorization using normal distributions

Approximates two matrices U,V s.t X \approx UV.
where U, V  have less dimensions than X
*/

data {
    int<lower=1> n_components; // Dimension of embeddings
    int<lower=0> n; // rows in data matrix
    int<lower=0> m; // columns in data matrix
    matrix[n,m] X; // data matrix

    real mu_u; // Prior mean of elements in U matrix
    real<lower=0> sigma_u; // Prior std of elements in U matrix

    real mu_v; // Prior mean of elements in V matrix
    real<lower=0> sigma_v; // Prior std of elemens in V matrix

    real<lower=0> sigma_x; // X ~ N(U*V, sigma_x) 
}

parameters {
    matrix[n, n_components] U;
    matrix[n_components, m] V;
}

model {
    matrix[n,m] X_hat;

    to_vector(U) ~ normal(mu_u, sigma_u);
    to_vector(V) ~ normal(mu_v, sigma_v);

    X_hat = U*V;

    for (i in 1:n) {
        X[i] ~ normal(X_hat[i], sigma_x);
    }
}