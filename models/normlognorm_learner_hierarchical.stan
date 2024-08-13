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
  int K=11;
}

parameters {
  real alpha_0_mu;
  real<lower=0> alpha_0_sigma;
  vector[M] alpha_0_z;
  
  real alpha_t_mu;
  real<lower=0> alpha_t_sigma;
  vector[M] alpha_t_z;
  
  real a_rt_mu;
  real<lower=0> a_rt_sigma;
  vector[M] a_rt_z;
  
  real a_acc_mu;
  real<lower=0> a_acc_sigma;
  vector[M] a_acc_z;
  
  vector[K] beta_rt_mu;
  real<lower=0> beta_rt_sigma;
  matrix[K, M] beta_rt_z;
  
  vector[K] beta_acc_mu;
  real<lower=0> beta_acc_sigma;
  matrix[K, M] beta_acc_z;
  
  vector<lower=0,upper=1>[M] ndt_raw;
  vector<lower=0>[M] sigma;
  vector[M] tau;
}

transformed parameters {
  vector[N] switchProp;
  vector[N] incProp;
  matrix[N, K] X;
  vector[M] ndt = RTmin .* ndt_raw;
  vector[M] alpha_0 = alpha_0_mu + alpha_0_sigma * alpha_0_z;
  vector[M] alpha_t = alpha_t_mu + alpha_t_sigma * alpha_t_z;
  vector[M] a_rt = a_rt_mu + a_rt_sigma * a_rt_z;
  vector[M] a_acc = a_acc_mu + a_acc_sigma * a_acc_z;
  matrix[K, M] beta_rt = rep_matrix(beta_rt_mu, M) + beta_rt_sigma .* beta_rt_z;
  matrix[K, M] beta_acc = rep_matrix(beta_acc_mu, M) + beta_acc_sigma .* beta_acc_z;

  for (t in 1:N) {
    if (trial[t] == 1) {
      switchProp[t] = 0.5;
      incProp[t] = 0.5;
    } else {
      real lr = inv_logit(alpha_0[S[t]] + alpha_t[S[t]] * trial[t-1]);
      switchProp[t] = switchProp[t-1] + lr * (isSwitch[t-1] - switchProp[t-1]);
      incProp[t] = incProp[t-1] + lr * (isInc[t-1] - incProp[t-1]);
    }
  }
  
  // switchProp = (switchProp - min(switchProp)) / (max(switchProp) - min(switchProp));
  // incProp = (incProp - min(incProp)) / (max(incProp) - min(incProp));
  switchProp = (switchProp-0.25)/.5;
  incProp = (incProp-0.25)/.5;
  
  X[, 1] = isInc;
  X[, 2] = isSwitch;
  X[, 3] = isInc .* isSwitch;
  X[, 4] = incProp;
  X[, 5] = switchProp;
  X[, 6] = incProp .* switchProp;
  X[, 7] = incProp .* isInc;
  X[, 8] = switchProp .* isSwitch;
  X[, 9] = isInc .* isSwitch .* incProp;
  X[, 10] = isInc .* isSwitch .* switchProp;
  X[, 11] = isInc .* isSwitch .* incProp .* switchProp;
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
      ndt_vec[t] = ndt[S[t]];
    }
    
    (RT - ndt_vec) ~ lognormal(eta_rt, sigma_vec);
    acc ~ bernoulli(Phi_approx(eta_acc));
  }
  
  alpha_0_mu ~ normal(0, 2.5);
  alpha_0_sigma ~ normal(0, 2.5);
  alpha_0_z ~ std_normal();
  
  alpha_t_mu ~ normal(0, 0.25);
  alpha_t_sigma ~ normal(0, 0.25);
  alpha_t_z ~ std_normal();
  
  a_rt_mu ~ normal(-1, 1.5);
  a_rt_sigma ~ normal(0, 1.5);
  a_rt_z ~ std_normal();
  
  a_acc_mu ~ normal(1.5, 1);
  a_acc_sigma ~ std_normal();
  a_acc_z ~ std_normal();
  
  beta_rt_mu ~ std_normal();
  beta_rt_sigma ~ std_normal();
  to_vector(beta_rt_z) ~ std_normal();
  
  beta_acc_mu ~ std_normal();
  beta_acc_sigma ~ std_normal();
  to_vector(beta_acc_z) ~ std_normal();
  
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
