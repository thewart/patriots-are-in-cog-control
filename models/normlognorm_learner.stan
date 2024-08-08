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
  vector<lower=0,upper=1>[N] isInc; //if trial is incongruent [1] or congruent [0]
  vector<lower=0,upper=1>[N] isSwitch; //if trial is switch [1] or repeat[1]
  real<lower=0> RTmin;
  array[N] int trial; //trial witin block
}

transformed data {
  int K=5;
}

parameters {
  real<lower=0> alpha_0;
  real alpha_t;

  real a_rt;
  real a_acc;
  vector[K] beta_rt;
  vector[K] beta_acc;
  
  real<lower=0,upper=RTmin> ndt;
  real<lower=0> sigma;
  real tau;
}

transformed parameters {
  vector[N] switchProp;
  vector[N] incProp;
  matrix[N, K] X;
  
  for (t in 1:N) {
    if (trial[t] == 1) {
      switchProp[t] = 0.5;
      incProp[t] = 0.5;
    } else {
      real lr = inv(alpha_0 + alpha_t*(trial[t-1]-1)/100 + 1)
      switchProp[t] = switchProp[t-1] + lr * (isSwitch[t-1] - switchProp[t-1]);
      incProp[t] = incProp[t-1] + lr * (isInc[t-1] - switchProp[t-1]);
    }
  }
  
  X[, 1] = isInc;
  X[, 2] = isSwitch;
  X[, 3] = isInc .* isSwitch;
  X[, 4] = abs(incProp - isInc);
  X[, 5] = abs(switchProp - isSwitch);
}

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

