/*
Matrix factorization using normal distributions

Approximates two matrices U,V s.t X \approx UV.
where U, V  have less dimensions than X
*/

data {
}

parameters {
    matrix[2,2] temp;
}

model {
    to_vector(temp) ~ normal(0,1);
    print("Temp")
    print(temp);
    print("Temp*Temp")
    print(temp*temp)
}