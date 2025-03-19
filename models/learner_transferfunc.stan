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
  int K=10;
  // real<lower=0> kappa=0;
}

parameters {
  real alpha_0;
  real alpha_t;
  real<lower=0> kappa;

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
  vector[N] switchPrep;
  vector[N] incPrep;
  matrix[N, K] X;
  
  for (t in 1:N) {
    if (trial[t] == 1) {
      switchProp[t] = 0.5;
      incProp[t] = 0.5;
    } else {
      real lr = inv_logit(alpha_0 + alpha_t * trial[t-1]);
      switchProp[t] = switchProp[t-1] + lr * (isSwitch[t-1] - switchProp[t-1]);
      incProp[t] = incProp[t-1] + lr * (isInc[t-1] - incProp[t-1]);
    }
  }
  
  switchPrep = inv_logit((switchProp-0.5)/.5 .* kappa);
  incPrep = inv_logit((incProp-0.5) .* kappa);
  
  X[, 1] = isInc;
  X[, 2] = isSwitch;
  X[, 3] = isInc .* isSwitch;
  X[, 4] = incPrep;
  X[, 5] = switchPrep;
  X[, 6] = incPrep .* isInc;
  X[, 7] = switchPrep .* isSwitch;
  X[, 8] = isInc .* isSwitch .* incPrep;
  X[, 9] = isInc .* isSwitch .* switchPrep;
  X[, 10] = isInc .* isSwitch .* incPrep .* switchPrep;
}

model {
  {
    vector[N] eta_rt = a_rt + X * beta_rt;
    vector[N] resid_rt = log(RT - ndt) - eta_rt;
    vector[N] eta_acc = a_acc + X * beta_acc + tau * resid_rt;

    (RT - ndt) ~ lognormal(eta_rt, sigma);
    acc ~ bernoulli(Phi_approx(eta_acc));
  }
  
  alpha_0 ~ normal(0, 2.5);
  alpha_t ~ normal(0, 0.1);

  a_rt ~ normal(1, 1);
  beta_rt ~ std_normal();
  beta_acc ~ std_normal();
  ndt ~ normal(0, 0.3);
  tau ~ std_normal();
  sigma ~ normal(0, 2);
  kappa ~ normal(2, 5);
}

generated quantities {

  vector[N] log_lik;
  real log_lik_total;
  {
    vector[N] eta_rt = a_rt + X * beta_rt;
    vector[N] resid_rt = log(RT - ndt) - eta_rt;
    vector[N] eta_acc = a_acc + X * beta_acc + tau * resid_rt;
    for (i in 1:N) log_lik[i] = lognormal_lpdf(RT[i] - ndt | eta_rt[i], sigma) + bernoulli_lpmf(acc[i] | Phi_approx(eta_acc[i]));
  }
  log_lik_total = sum(log_lik);
}
