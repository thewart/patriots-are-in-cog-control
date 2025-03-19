functions {
  matrix design_matrix(vector isInc, vector isSwitch, vector controlLevel, int K) {
    int N = num_elements(isInc);
    matrix[N, K] X;
    
    X[, 1] = isInc;
    X[, 2] = isSwitch;
    X[, 3] = isInc .* isSwitch;
    X[, 4] = controlLevel;
    X[, 5] = controlLevel .* isInc;
    X[, 6] = controlLevel .* isSwitch;
    X[, 7] = controlLevel .* isInc .* isSwitch;
    return X;
  }
  
  row_vector etafy(real RT, vector a, matrix beta, row_vector X, real tau, real ndt) {
    row_vector[2] eta = a' + X * beta;
    eta[2] = eta[2] + tau * (log(RT - ndt) - eta[1]);
    return eta;
  }
  
  real RR_rng(vector a, matrix beta, row_vector X, real tau, real ndt, real sigma) {
    real RT_pp;
    real acc_pp;
    real RT_resid = sigma * std_normal_rng();
    row_vector[2] eta = a' + X * beta;
    eta[2] = eta[2] + tau * RT_resid;
    RT_pp = ndt + exp(eta[1] + RT_resid);
    acc_pp = Phi_approx(eta[2]);

    return acc_pp/RT_pp;
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
  int K = 7;
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
  // simplex[K] beta_row_sigma;
  vector<lower=0>[2] beta_col_sigma;
  cholesky_factor_corr[2] beta_col_L;
  array[M] matrix[K, 2] beta_z;
  
  real<lower=0> gamma_switch_mu;
  real gamma_inc_mu;
  real gamma_0_mu;
  vector<lower=0>[3] gamma_sigma;
  array[M] vector[3] gamma_z;
  
  vector<lower=0,upper=1>[M] ndt_raw;
  vector<lower=0>[M] sigma;
  vector[M] tau;
}

transformed parameters {
  vector[N] switchProp;
  vector[N] incProp;
  vector[N] controlLevel;
  matrix[N, K] X;
  vector[M] alpha_0 = alpha_0_mu + alpha_0_sigma * alpha_0_z;
  // vector[M] alpha_t = alpha_t_mu + alpha_t_sigma * alpha_t_z;
  matrix[2, M] a = rep_matrix(a_mu, M) + diag_pre_multiply(a_sigma, a_L) * a_z;
  vector[3] gamma_mu = [gamma_0_mu, gamma_inc_mu, gamma_switch_mu]';
  array[M] vector[3] gamma;
  array[M] matrix[K, 2] beta;
  vector[M] ndt = RTmin .* ndt_raw;

  {
    matrix[2, 2] VT = diag_pre_multiply(beta_col_sigma, beta_col_L');
    for (j in 1:M) beta[j] = beta_mu + diag_pre_multiply(rep_vector(1, K), beta_z[j] * VT);
  }
  
  for (j in 1:M) {
    gamma[j] = gamma_mu + gamma_sigma .* gamma_z[j];
  }
  
  for (t in 1:N) {
    if (trial[t] == 1) {
      switchProp[t] = 0.5;
      incProp[t] = 0.5;
    } else {
      real lr = inv_logit(alpha_0[S[t]]); //+ alpha_t[S[t]] * trial[t-1]);
      switchProp[t] = switchProp[t-1] + lr * (isSwitch[t-1] - switchProp[t-1]);
      incProp[t] = incProp[t-1] + lr * (isInc[t-1] - incProp[t-1]);
    }
  }
  
  
  switchProp = (switchProp-0.5)/.5;
  incProp = (incProp-0.5)/.5;
  // for (t in 1:N) controlLevel[t] = inv_logit(gamma[S[t]][1] + gamma[S[t]][2]*incProp[t] + gamma[S[t]][3]*switchProp[t]);
  for (t in 1:N) controlLevel[t] = inv_logit([1, incProp[t], switchProp[t]] * gamma[S[t]]);
  X = design_matrix(isInc, isSwitch, 2*(controlLevel-.5), K);

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
  
  gamma_mu ~ normal(0, 2.5);
  gamma_sigma ~ normal(0, 2.5);
  for (j in 1:M) gamma_z[j] ~ std_normal();
  
  ndt ~ normal(0, 0.3);
  target += sum(log(RTmin));
  tau ~ std_normal();
  sigma ~ normal(0, 2);
}

generated quantities {
  real a_rho = (a_L * a_L')[1, 2];
  real beta_rho = (beta_col_L * beta_col_L')[1, 2];
  vector[N] log_lik;
  array[N] real RR_pp;
  array[9,9] vector[4] RR_mu;

  for (t in 1:N) {
    row_vector[2] eta = etafy(RT[t], col(a, S[t]), beta[S[t]], X[t], tau[S[t]], ndt[S[t]]);
    log_lik[t] = lognormal_lpdf(RT[t] - ndt[S[t]] | eta[1], sigma[S[t]]) + bernoulli_lpmf(acc[t] | Phi_approx(eta[2]));
    
    RR_pp[t] = RR_rng(col(a, S[t]), beta[S[t]], X[t], tau[S[t]], ndt[S[t]], sigma[S[t]]);
  }
  
  {
    vector[9] incProp_rep = [-1.0, -.75, -.5, -.25, 0, .25, .5, .75, 1.0]';
    vector[9] switchProp_rep = [-1.0, -.75, -.5, -.25, 0, .25, .5, .75, 1.0]';
    real mu_sigma = mean(sigma);
    real mu_tau = mean(tau);
    real mu_ndt = mean(ndt);
    vector[4] isInc_rep = [0, 1, 0, 1]';
    vector[4] isSwitch_rep = [0, 0, 1, 1]';
    for (i in 1:9) {
      for (j in 1:9) {
        real ctrl_tmp = inv_logit([1, incProp_rep[i], switchProp_rep[j]] * gamma_mu);
        matrix[4, K] X_tmp = design_matrix(isInc_rep, isSwitch_rep, rep_vector(ctrl_tmp, 4), K);
        for (t in 1:4) RR_mu[i,j][t] = RR_rng(a_mu, beta_mu, X_tmp[t], mu_tau, mu_ndt, mu_sigma);
      }
    }
  }
}
  
