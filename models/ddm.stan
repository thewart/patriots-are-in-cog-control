functions {
    real wiener_diffusion_lpdf(real y, int dec, real alpha,
                              real tau, real beta, real delta) {
    if (dec == 1) {
      return wiener_lpdf(y | alpha, tau, beta, delta);
    } else {
      return wiener_lpdf(y | alpha, tau, 1 - beta, - delta);
    }
  }
 
  real get_wiener_lpdf_sum(vector RTs, array[] int acc,
                          vector bs, vector ndt, vector bias, vector drift) {
    int N = num_elements(RTs);
    vector[N] lpdf;
    
    for (n in 1:N){
      lpdf[n] = wiener_diffusion_lpdf(RTs[n] | acc[n], bs[n], ndt[n], bias[n], drift[n]);
    }
    
    return sum(lpdf);
  }
}

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
  real a_d;  // drift
  real a_bs; // boundary separation
  real<lower=0,upper=RTmin> a_ndt; // non decision time
  vector[K] b_d;  // drift betas
  vector[K] b_bs;  // boundary separation betas
  // vector[K] b_ndt;  // non decsion time betas
}

// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {
  {
    vector[N] drift = a_d + X * b_d;
    vector[N] bs = exp(a_bs + X * b_bs);
    vector[N] ndt = rep_vector(a_ndt, N);
    vector[N] bias = rep_vector(0.5, N);
    
    target += get_wiener_lpdf_sum(RT, acc, bs, ndt, bias, drift);
  }
  
  target += normal_lpdf(a_d | 2, 0.5);
  target += normal_lpdf(a_bs | .5, .2);
  target += normal_lpdf(b_d | 0, 0.2);
  target += normal_lpdf(b_bs | 0, 0.2);
  a_ndt ~ normal(0, 0.3);
  
}

