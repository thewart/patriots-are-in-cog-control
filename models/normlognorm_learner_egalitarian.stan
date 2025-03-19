data {
  int<lower=1> N;  // total number of observations
  int<lower=1> M; // total number of subjects
  vector[N] RT;  // RTs on each trial
  array[N] int acc;  // accuracy on each trial
  vector<lower=0,upper=1>[N] isInc; //if trial is incongruent [1] or congruent [0]
  vector<lower=0,upper=1>[N] isSwitch; //if trial is switch [1] or repeat[1]
  vector[M] RTmin;
  array[N] int trial; //trial witin block
  array[N] int S;
}

transformed data {
  int K=10;
}

parameters {
  vector[M] alpha_0;
  vector[M] alpha_t;
  // real<lower=0,upper=1> alpha_asym;

  vector[M] a_rt;
  vector[M] a_acc;
  matrix[K, M] beta_rt;
  matrix[K, M] beta_acc;
  
  vector[M]<lower=0,upper=1> ndt_raw;
  vector[M]<lower=0> sigma;
  vector[M] tau;
}

transformed parameters {
  vector[N] switchProp;
  vector[N] incProp;
  matrix[N, K] X;
  vector[N] ndt = RTmin .* ndt_raw;

  for (t in 1:N) {
    if (trial[t] == 1) {
      switchProp[t] = 0.5;
      incProp[t] = 0.5;
    } else {
      real lr = inv_logit(alpha_0[S[t]] + alpha_t[S[t]] * trial[t-1]); //alpha_asym;
      switchProp[t] = switchProp[t-1] + lr * (isSwitch[t-1] - switchProp[t-1]);
      incProp[t] = incProp[t-1] + lr * (isInc[t-1] - incProp[t-1]);
    }
  }
  
  switchProp = (switchProp - min(switchProp)) / (max(switchProp) - min(switchProp));
  incProp = (incProp - min(incProp)) / (max(incProp) - min(incProp));
  
  X[, 1] = isInc;
  X[, 2] = isSwitch;
  X[, 3] = isInc .* isSwitch;
  X[, 4] = incProp;
  X[, 5] = switchProp;
  X[, 6] = incProp .* isInc;
  X[, 7] = switchProp .* isSwitch;
  X[, 8] = isInc .* isSwitch .* incProp;
  X[, 9] = isInc .* isSwitch .* switchProp;
  X[, 10] = isInc .* isSwitch .* incProp .* switchProp;
}

model {
  {
    vector[N] eta_rt;
    vector[N] resid_rt;
    vector[N] eta_acc;
    vector[N] sigma_vec;
    vector[N] ndt_vec;
    for (t in 1:N) {
      eta_rt[t] = a_rt[S[t]] + X[t] * col(beta_rt, S[t]);
      resid_rt[t] = log(RT[t] - ndt[S[t]]) - eta_rt[t];
      eta_acc[t] = a_acc[S[t]] + X[t] * col(beta_acc, S[t]) + tau[S[t]] * resid_rt[t];
      sigma_vec[t] = sigma[S[t]];
      ndt_vec[t] = ndt
    }
    
    (RT - ndt_vec) ~ lognormal(eta_rt, sigma_vec);
    acc ~ bernoulli(Phi_approx(eta_acc));
  }
  
  alpha_0 ~ normal(0, 2.5);
  alpha_t ~ normal(0, 0.5);

  a_rt ~ normal(-0.5, 2.5);
  a_acc ~ normal(1.5, 1);
  to_vector(beta_rt) ~ std_normal();
  to_vector(beta_acc) ~ std_normal();
  ndt ~ normal(0, 0.3);
  target += sum(log(RTmin));
  tau ~ std_normal();
  sigma ~ normal(0, 2);
}

generated quantities {
  
  vector[N] log_lik;
  real log_lik_total;
  {
    real eta_rt;
    real resid_rt;
    real eta_acc;
    for (t in 1:N) {
      eta_rt = a_rt[S[t]] + X[t] * col(beta_rt, S[t]);
      resid_rt = log(RT[t] - ndt[S[t]]) - eta_rt;
      eta_acc = a_acc[S[t]] + X[t] * col(beta_acc, S[t]) + tau[S[t]] * resid_rt;
      log_lik[t] = lognormal_lpdf(RT[t] - ndt[S[t]] | eta_rt, sigma[S[t]]) + bernoulli_lpmf(acc[t] | Phi_approx(eta_acc));
    }
  }
  log_lik_total = sum(log_lik);
}
