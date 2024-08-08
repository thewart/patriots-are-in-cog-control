datadir <- "../modeling-of-control/data/"

# rstan options
rstan_options(auto_write = TRUE)

#load data
df <- read.csv(paste0(datadir, "dataset1.csv")) %>% 
  dplyr::select(trialCount, block, acc, RT, stimCongruency, switchType, subject) %>% 
  dplyr::mutate(RT = RT/1000,
         stimCongruency = ifelse(stimCongruency == "i", 1, 0),
         switchType = ifelse(switchType == "s", 1, 0),
         subject = match(subject, unique(subject))) %>% 
  dplyr::filter(RT < 1.5 & RT > 0.3) |> as.data.table()

stan_data <- with(df[subject==9], {
  list(X=model.matrix(~ switchType*stimCongruency)[,-1],
       RT=RT, acc=acc, N=length(RT), K=3, RTmin=min(RT), )
})
stan_data$trial = 

model_normlognorm <- stan_model("models/normlognorm.stan")
fit_normlognorm <- sampling(model_normlognorm, stan_data, iter=400)

model_ddm <- stan_model("models/ddm.stan")
fit_ddm <- sampling(model_ddm, stan_data, iter=400)

model_normlognorm_learner <- stan_model("models/normlognorm_learner.stan")

##### raw tradeoffs
foo <- df[, mean(RT), by=.(stimCongruency, incProp, subject)] |> dcast(subject ~ stimCongruency + incProp)
foo2 <- df[, mean(RT), by=.(switchType, switchProp, subject)] |> dcast(subject ~ switchType + switchProp)