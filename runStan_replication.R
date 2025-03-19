source("helper_funcs.R")
parstokeep <- c("alpha_0_mu", "alpha_0_sigma", "a_mu", "a_sigma", "beta_mu", "beta_col_sigma", "beta_row_sigma",
                "a_rho", "beta_rho", "ndt", "tau", "sigma", "log_lik", "RR_pp") 

df <- read_dataset_2(datadir)
stan_data <- get_stan_data(df)

model_2d_full <- stan_model("models/normlognorm_full2D.stan")
fit_2d_full <- sampling(model_2d_full, stan_data, iter=500, warmup=200, chains=4, pars=c(parstokeep, "RR_mu"), init_r=0.5)
save(fit_2d_full, file="fit_2d_full_rep.Rdat")

model_2d_narrow <- stan_model("models/normlognorm_narrow2D.stan")
fit_2d_narrow <- sampling(model_2d_narrow, stan_data, iter=500, warmup=200, chains=4, pars=parstokeep, init_r=0.5)
save(fit_2d_narrow, file="fit_2d_narrow_rep.Rdat")

model_1d_simple <- stan_model("models/normlognorm_1D_simple.stan")
fit_1d_simple <- sampling(model_1d_simple, stan_data, iter=500, warmup=200, chains=4,
                   pars=c(parstokeep, "omega_mu", "omega_sigma", "omega", "beta"), init_r=0.5)
save(fit_1d_simple, file=paste0("fit_1d_simple_rep.Rdat"))

model_1d <- stan_model("models/normlognorm_1D_forceconstraint.stan")
fit_1d <- sampling(model_1d, stan_data, iter=500, warmup=200, chains=4,
                   pars=c(parstokeep, "gamma", "gamma_mu", "gamma_sigma", "beta"), init_r=0.5)
save(fit_1d, file=paste0("fit_1d_rep.Rdat"))




