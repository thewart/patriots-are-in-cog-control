//
// This Stan program defines a simple model, with a
// vector of values 'y' modeled as normally distributed
// with mean 'mu' and standard deviation 'sigma'.
//
// Learn more about model development with Stan at:
//
//    http://mc-stan.org/users/interfaces/rstan.html
//    https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started
//

// The input data is a vector 'y' of length 'N'.
data {
  int<lower=1> N;  // total number of observations
  vector[N] RT;  // RTs on each trial
  array[N] int acc;  // accuracy on each trial
  int<lower=1> K; // number of fixed-effects
  matrix[N, K] X; // main effects
  real<lower=0> RTmin;
}


// The parameters accepted by the model. Our model
// accepts two parameters 'mu' and 'sigma'.
parameters {
  real a_rt;
  real a_acc;
  vector[K] beta_rt;
  vector[K] beta_acc;
  
  real<lower=0,upper=RTmin> ndt;
  real<lower=0> sigma;
  real tau;
}

// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {
  {
    vector[N] eta_rt = a_rt + X * beta_rt;
    vector[N] resid_rt = log(RT - ndt) - eta_rt;
    vector[N] eta_acc = a_acc + X * beta_acc + tau * resid_rt;
    
    // resid_rt ~ normal(0, sigma);
    // target += sum(-log(RT - ndt));
    (RT - ndt) ~ lognormal(eta_rt, sigma);
    acc ~ bernoulli(Phi_approx(eta_acc));
  }
  
  a_rt ~ normal(1, 1);
  beta_rt ~ std_normal();
  beta_acc ~ std_normal();
  ndt ~ normal(0, 0.3);
  tau ~ std_normal();
  sigma ~ normal(0, 2);
}

