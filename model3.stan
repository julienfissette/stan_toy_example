data {
  int<lower=0> N;
  vector[N] x;
  vector[N] y;
  int factor[N];

  int no_factors; # will still be 8

  int N_pred;
  int factor_pred[N_pred];
  vector[N_pred] x_pred;
}

parameters {
  vector[no_factors] alpha;
  vector[no_factors] beta;
  real<lower=0> sigma;
  real grandalpha;
  real grandbeta;
}
transformed parameters{
    vector[N] meanvector;

    for(n in 1:N){
        meanvector[n] = alpha[factor[n]] + beta[factor[n]] * x[n];
    }
}
model {
  grandalpha ~ normal(0, 10);
  grandbeta ~ normal(0, 5);

  beta ~ normal(grandbeta, 3);
  alpha ~ normal(grandalpha, 3);

  y ~ normal(meanvector, sigma);
}
generated quantities{
    vector[N_pred] meanvector_pred;
    vector[N_pred] y_pred;

    for(n in 1:N_pred){
        meanvector_pred[n] = alpha[factor_pred[n]] + beta[factor_pred[n]] * x_pred[n];
    }

   y_pred = to_vector(normal_rng(meanvector_pred, sigma));
}
