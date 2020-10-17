data {
  int<lower=0,upper=1000> N; # number of samples
  vector[N] x; # vector
  vector[N] y;
}
parameters {
  real alpha;
  real beta;
  real<lower=0> sigma;
}
transformed parameters{
    vector[N] meanvector = alpha + beta * x;
}
model {

  beta ~ normal(0,5);
  alpha ~ normal(0,10);

  y ~ normal(meanvector, sigma);
}
generated quantities{
  real cum_sum;
  cum_sum = sum(y);
}