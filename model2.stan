data {
  int<lower=0> N;
  vector[N] x;
  vector[N] y;
  int no_factors;
  int factor[N];
}
parameters {
  vector[no_factors] alpha;
  vector[no_factors] beta;
  real grandalpha;
  real grandbeta;
  real<lower=0> sigma;
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
