datadir <- "../modeling-of-control/data/"
parstokeep <- c("alpha_0_mu", "alpha_0_sigma", "alpha_t_mu", "alpha_t_sigma",
                "a_mu", "a_sigma", "beta_mu", "beta_col_sigma", "beta_row_sigma", "a_rho", "beta_rho",
                "ndt", "sigma", "tau", "log_lik")
# rstan options
rstan_options(auto_write = TRUE)

#load data
df <- read.csv(paste0(datadir, "dataset1.csv")) |> 
  dplyr::select(trialCount, block, acc, RT, stimCongruency, switchType, subject) |> 
  dplyr::mutate(RT = RT/1000,
         stimCongruency = ifelse(stimCongruency == "i", 1, 0),
         switchType = ifelse(switchType == "s", 1, 0),
         subject = match(subject, unique(subject))) |> 
  dplyr::filter(RT < 1.5 & RT > 0.3) |> as.data.table()
df[,trial := 1:.N, by=.(subject, block)]

stan_data <- with(df[subject==1], {
  list(X=model.matrix(~ switchType*stimCongruency)[,-1],
       RT=RT, acc=acc, N=length(RT), K=3, RTmin=min(RT))
})

model_normlognorm <- stan_model("models/normlognorm.stan")
fit_normlognorm <- sampling(model_normlognorm, stan_data, iter=400)

model_ddm <- stan_model("models/ddm.stan")
fit_ddm <- sampling(model_ddm, stan_data, iter=400)

model_learner <- stan_model("models/normlognorm_learner.stan")
stan_data <- with(df[subject==19], list(isInc=stimCongruency, isSwitch=switchType, RT=RT, acc=acc,
                                       N=length(RT), K=3, RTmin=min(RT), trial=trial))
fit_learner_5 <- sampling(model_learner, stan_data, iter=400, pars=c("X", "switchProp", "incProp", "log_lik"), include=F, refresh=F)

ds <- df
stan_data <- list(isInc=ds$stimCongruency, isSwitch=ds$switchType, RT=ds$RT, acc=ds$acc,
                  N=nrow(ds), M=uniqueN(ds$subject), K=3, RTmin=ds[,min(RT), by=subject]$V1,
                  trial=ds$trial, S=ds$subject)
model_hier <- stan_model("models/normlognorm_learner_hierarchical_allin.stan")
fit_hier <- sampling(model_hier, stan_data, iter=10, chains=1, pars=parstokeep, init="0")


##### raw tradeoffs
foo <- df[, mean(RT), by=.(stimCongruency, incProp, subject)] |> dcast(subject ~ stimCongruency + incProp)
foo2 <- df[, mean(RT), by=.(switchType, switchProp, subject)] |> dcast(subject ~ switchType + switchProp)