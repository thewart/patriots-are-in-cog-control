functions {
  matrix design_matrix(vector isInc, vector isSwitch, vector incProp, vector switchProp, int K) {
    int N = num_elements(isInc);
    matrix[N, K] X;
    
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
    return X;
  }
  
  row_vector etafy(real RT, vector a, matrix beta, row_vector X, real tau, real ndt) {
    row_vector[2] eta = a' + X * beta;
    eta[2] = eta[2] + tau * (log(RT - ndt) - eta[1]);
    return eta;
  }
}

data {
  int<lower=1> N;  // total number of observations
  int<lower=1> M; // total number of subjects
  // int K;
  vector[N] RT;  // RTs on each trial
  array[N] int acc;  // accuracy on each trial
  vector<lower=0,upper=1>[N] isInc; //if trial is incongruent [1] or congruent [0]
  vector<lower=0,upper=1>[N] isSwitch; //if trial is switch [1] or repeat[1]
  vector[M] RTmin;
  array[N] int trial; //trial witin block
  array[N] int S;
  // int<lower=1> N_rep;
  // matrix[N_rep, K] X_rep;
}

transformed data {
  int K = 11;
}

parameters {
  real alpha_0_mu;
  real<lower=0> alpha_0_sigma;
  vector[M] alpha_0_z;
  
  // real alpha_t_mu;
  // real<lower=0> alpha_t_sigma;
  // vector[M] alpha_t_z;
  
  vector[2] a_mu;
  vector<lower=0>[2] a_sigma;
  cholesky_factor_corr[2] a_L;
  matrix[2, M] a_z;
  
  matrix[K, 2] beta_mu;
  simplex[K] beta_row_sigma;
  vector<lower=0>[2] beta_col_sigma;
  cholesky_factor_corr[2] beta_col_L;
  array[M] matrix[K, 2] beta_z;
  
  vector<lower=0,upper=1>[M] ndt_raw;
  vector<lower=0>[M] sigma;
  vector[M] tau;
}

transformed parameters {
  vector[N] switchProp;
  vector[N] incProp;
  matrix[N, K] X;
  vector[M] alpha_0 = alpha_0_mu + alpha_0_sigma * alpha_0_z;
  // vector[M] alpha_t = alpha_t_mu + alpha_t_sigma * alpha_t_z;
  matrix[2, M] a = rep_matrix(a_mu, M) + diag_pre_multiply(a_sigma, a_L) * a_z;
  array[M] matrix[K, 2] beta;
  vector[M] ndt = RTmin .* ndt_raw;
  
  {
    matrix[2, 2] VT = diag_pre_multiply(beta_col_sigma, beta_col_L');
    for (j in 1:M) beta[j] = beta_mu + diag_pre_multiply(beta_row_sigma, beta_z[j] * VT);
  }
  
  for (t in 1:N) {
    if (trial[t] == 1) {
      switchProp[t] = 0.5;
      incProp[t] = 0.5;
    } else {
      real lr = inv_logit(alpha_0[S[t]]); // + alpha_t[S[t]] * trial[t-1]);
      switchProp[t] = switchProp[t-1] + lr * (isSwitch[t-1] - switchProp[t-1]);
      incProp[t] = incProp[t-1] + lr * (isInc[t-1] - incProp[t-1]);
    }
  }
  
  switchProp = (switchProp-0.5)/.5;
  incProp = (incProp-0.5)/.5;
  X = design_matrix(isInc, isSwitch, incProp, switchProp, K);

}

model {
  {
    matrix[N, 2] eta;
    vector[N] sigma_vec;
    vector[N] ndt_vec;
    for (t in 1:N) {
      eta[t] = etafy(RT[t], col(a, S[t]), beta[S[t]], X[t], tau[S[t]], ndt[S[t]]);
      sigma_vec[t] = sigma[S[t]];
      ndt_vec[t] = ndt[S[t]];
    }
    
    (RT - ndt_vec) ~ lognormal(col(eta, 1), sigma_vec);
    acc ~ bernoulli(Phi_approx(col(eta, 2)));
  }
  
  alpha_0_mu ~ normal(0, 2.5);
  alpha_0_sigma ~ normal(0, 2.5);
  alpha_0_z ~ std_normal();
  
  // alpha_t_mu ~ normal(0, 0.25);
  // alpha_t_sigma ~ normal(0, 0.25);
  // alpha_t_z ~ std_normal();
  
  a_mu ~ normal([-1, 1.5]', [1.5, 1]');
  a_sigma ~ std_normal();
  to_vector(a_z) ~ std_normal();
  
  to_vector(beta_mu) ~ std_normal();
  beta_col_sigma ~ normal(0, 2.5);
  for (j in 1:M) to_vector(beta_z[j]) ~ std_normal();
  
  ndt ~ normal(0, 0.3);
  target += sum(log(RTmin));
  tau ~ std_normal();
  sigma ~ normal(0, 2);
}

generated quantities {
  real a_rho = (a_L * a_L')[1, 2];
  real beta_rho = (beta_col_L * beta_col_L')[1, 2];
  vector[N] log_lik;
  array[N] real RT_pp;
  array[N] real acc_pp;
  
  for (t in 1:N) {
    row_vector[2] eta = etafy(RT[t], col(a, S[t]), beta[S[t]], X[t], tau[S[t]], ndt[S[t]]);
    real RT_resid = sigma[S[t]] * std_normal_rng();
    
    log_lik[t] = lognormal_lpdf(RT[t] - ndt[S[t]] | eta[1], sigma[S[t]]) + bernoulli_lpmf(acc[t] | Phi_approx(eta[2]));
    RT_pp[t] = ndt[S[t]] + exp(eta[1] + RT_resid);
    acc_pp[t] = Phi_approx(col(a, S[t])[2] + X[t] * col(beta[S[t]], 2) + tau[S[t]] * RT_resid);
  }
}
  
