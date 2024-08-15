datadir <- "../modeling-of-control/data/"
parstokeep <- c("alpha_0_mu", "alpha_0_sigma", "alpha_t_mu", "alpha_t_sigma",
                "a_mu", "a_sigma", "beta_mu", "beta_col_sigma", "beta_row_sigma", "a_rho", "beta_rho",
                "ndt", "sigma", "tau", "log_lik")

df <- read.csv(paste0(datadir, "dataset1.csv")) |> 
  dplyr::select(trialCount, block, acc, RT, stimCongruency, switchType, subject) |> 
  dplyr::mutate(RT = RT/1000,
                stimCongruency = ifelse(stimCongruency == "i", 1, 0),
                switchType = ifelse(switchType == "s", 1, 0),
                subject = match(subject, unique(subject))) |> 
  dplyr::filter(RT < 1.5 & RT > 0.3) |> as.data.table()
df[,trial := 1:.N, by=.(subject, block)]

stan_data <- list(isInc=df$stimCongruency, isSwitch=df$switchType, RT=df$RT, acc=df$acc,
                  N=nrow(df), M=uniqueN(df$subject), K=3, RTmin=df[,min(RT), by=subject]$V1,
                  trial=df$trial, S=df$subject)
model_hier <- stan_model("models/normlognorm_learner_hierarchical_allin.stan")
fit_hier <- sampling(model_hier, stan_data, iter=10, chains=1, pars=parstokeep, init="0")
save(fit_hier, paste0("fit_", Sys.time() |> as.numeric()))