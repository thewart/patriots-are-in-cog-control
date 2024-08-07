// The input data is a vector 'y' of length 'N'.
data {
  int<lower=1> N;  // total number of observations
  array[N] blockN; //trial witin block
  vector[N] RT;  // RTs on each trial
  array[N] int acc;  // accuracy on each trial
  vector<lower=0,upper=1>[N] isInc; //if trial is incongruent [1] or congruent [0]
  vector<lower=0,upper=1>[N] isSwitch; //if trial is switch [1] or repeat[1]
  real<lower=0> RTmin;
}

transformed data {
  int K = 11;
}

parameters {
  real alpha_0;
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
  vector switchProp[N];
  vector incProp[N];
  matrix X[N, K];
  
  for (t in 1:N) {
    if (blockN[t] == 1) {
      switchProp[t] = 0.5;
      incProp[t] = 0.5;
    } else {
      switchProp[t] = switchProp + inv(alpha_0 + alpha_t*(blockN[t-1]-1) + 1) * isSwitch;
      incProp[t] = incProp + inv(alpha_0 + alpha_t*(blockN[t-1]-1) + 1) * isInc;
    }
  }
  
  X[, 1] = isInc;
  X[, 2] = isSwitch;
  X[, 3] = switchProp;
  X[, 4] = incProp;
  
  // interactions
  X[, 5] = X[, 1] .* X[, 2]; // congruency vs task sequence
  X[, 6] = X[, 1] .* X[, 3]; // congruency vs switch prop
  X[, 7] = X[, 1] .* X[, 4]; // congruency vs inc prop
  X[, 8] = X[, 2] .* X[, 3]; // task sequence vs switch prop
  X[, 9] = X[, 2] .* X[, 4]; // task sequence vs inc prop
  X[, 10] = X[, 3] .* X[, 4]; // switch prop vs inc prop
  X[, 11] = X[, 1] .* X[, 2] .* X[, 3];
  X[, 12] = X[, 1] .* X[, 2] .* X[, 4];
  X[, 13] = X[, 1] .* X[, 3] .* X[, 4];
  X[, 14] = X[, 2] .* X[, 3] .* X[, 4];
  X[, 15] = X[, 1] .* X[, 2] .* X[, 3] .* X[, 4];
}

model {
  {
    vector[N] eta_rt = a_rt + X * beta_rt;
    vector[N] resid_rt = log(RT - ndt) - eta_rt;
    vector[N] eta_acc = a_acc + X * beta_acc + tau * resid_rt;
    
    target += lognormal_pdf(RT - ndt | eta_rt, sigma);
    acc ~ binomial(Phi_approx(eta_acc));
  }
}

