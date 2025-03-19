library(rstan)
library(loo)
library(tidyverse)
library(tidybayes)
library(data.table)

datadir <- "../modeling-of-control/data/"

cellwise_posterior <- function(draws, df) {
  draws <- draws[df[, .(t=1:.N, subject, congruency, taskSequence, switchProp, incProp)], on="t"]
  
  post <- draws[, .(RR_pp=mean(RR_pp)), by=.(.draw, congruency, taskSequence, switchProp, incProp)][
    , .(RT=mean(RR_pp), .lower=quantile(RR_pp, .025), .upper=quantile(RR_pp, .975)), by=.(congruency, taskSequence, switchProp, incProp)]
  setkey(post, congruency, taskSequence, switchProp, incProp)
  return(post)
}

coef_posterior <- function(draws, df, rename_list, include_in_plot_list, label="") {
  draws <- draws[df[, .(t=1:.N, subject, congruency, taskSequence, switchProp, incProp)], on="t"]
  juh <- draws[, lm(RR_pp ~ 1 + congruency*taskSequence*switchProp*incProp)$coef %>% list(variable=names(.), value=.), by=.draw][
    , .(.value=mean(value), .upper=quantile(value, 0.975), .lower=quantile(value, 0.0275)), by=variable]
  setnames(juh, "variable", "effect")
  juh$effect <- names(rename_list)
  juh <- juh[effect %in% include_in_plot_list]
  juh$effect <- factor(juh$effect, levels = rev(include_in_plot_list))
  juh$model <- label
  return(juh)
}

getX_2dfull <- function(isInc = c(0, 1, 0, 1), isSwitch = c(0, 0, 1, 1), incProp, switchProp) {
  X <- matrix(NA, nrow=4, ncol=15)
  X[, 1] = isInc;
  X[, 2] = isSwitch;
  X[, 3] = isInc * isSwitch;
  X[, 4] = incProp;
  X[, 5] = switchProp;
  X[, 6] = incProp * switchProp;
  X[, 7] = isInc * incProp;
  X[, 8] = isInc * switchProp;
  X[, 9] = isInc * incProp * switchProp;
  X[, 10] = isSwitch * switchProp;
  X[, 11] = isSwitch * incProp;
  X[, 12] = isSwitch * switchProp * incProp;
  X[, 13] = isInc * isSwitch * incProp;
  X[, 14] = isInc * isSwitch * switchProp;
  X[, 15] = isInc * isSwitch * incProp * switchProp;
  
  return(X)
}

simpost <- function(a, beta, X, ndt, sigma, tau, nsim=1e4) {
  if (is.vector(beta))
    beta <- matrix(beta, nrow=2)
  
  rtmu <- a[1] + X %*% beta[1,]
  rtsim_log <- rnorm(nsim*4, rtmu, sigma) |> matrix(nrow=4)
  rtsim <- exp(rtsim_log) + ndt
  residsim <- sweep(rtsim_log, 1, rtmu)
  
  accmu <- a[2] + X %*% beta[2,]
  accmu <- sweep(residsim*tau, 1, accmu, "+")
  accsim <- rnorm(nsim*4, accmu, 1) |> pnorm() |> matrix(nrow=4)
  
  return(apply(accsim/rtsim, 1, mean))
}

simpost_wrap <- function(dt, incProp, switchProp, isInc = c(0, 1, 0, 1), isSwitch = c(0, 0, 1, 1)) {
  a <- dt[1, c(a_mu.1, a_mu.2)]
  ndt <- dt[1, ndt]
  sigma <- dt[1, sigma]
  tau <- dt[1, tau]
  X <- getX_2dfull(isInc, isSwitch, incProp, switchProp)
  simout <- simpost(a, dt$beta, X, ndt, sigma, tau)
  
  return(data.table(crr=simout, isInc=isInc, isSwitch=isSwitch, incProp=incProp, switchProp=switchProp))
}

getbaselinecomps <- function(stanfit, vals=c(-.5,.5), SDoffset=0) {
  hyperpars <- spread_draws(fit_2d_full, ndt[s], sigma[s], tau[s])
  hyperpars <- setDT(hyperpars)[, .(ndt=mean(ndt), sigma=mean(sigma), tau=mean(tau)), by=.draw]
  mua <- spread_draws(fit_2d_full, a_mu[..]) |> setDT()
  mubeta <- spread_draws(fit_2d_full, beta_mu[i,j]) |> setDT()
  
  if (SDoffset!=0) {
    juh <- gather_draws(fit_2d_full, beta_row_sigma[i], beta_col_sigma[j]) |> setDT()
    juh <- juh[,.(j=1:2, .SD[.variable=="beta_row_sigma" & i==6, .value] * .SD[.variable=="beta_col_sigma", .value]),by=.draw]
    mubeta <- spread_draws(fit_2d_full, beta_mu[i,j]) |> setDT()
    mubeta[i==6 & j==2, beta_mu := beta_mu - SDoffset*juh[j==2, V2]]
    mubeta[i==6 & j==1, beta_mu := beta_mu + SDoffset*juh[j==1, V2]]
  }
  
  foo <- mubeta[mua[,.(a_mu.1, a_mu.2, .draw)], on=".draw"][hyperpars, on=".draw"]
  vals <- expand.grid(vals, vals)
  allpostsims <- data.table()
  for (i in 1:nrow(vals)) {
    v1 <- vals[i,1]
    v2 <- vals[i,2]
    allpostsims <- rbind(
      allpostsims,
      foo[, simpost_wrap(.SD, v1, v2), by=.draw]
    )
  }
  allpostsims[, trialType:=rep(c("Baseline", "Inc", "Switch", "IncxSwitch"), .N/4)]
  postsims_wide <- dcast(allpostsims, .draw + incProp + switchProp ~ trialType, value.var = "crr")
  
  return(postsims_wide)
}

read_dataset_1 <- function(datadir) {
  df <- read.csv(paste0(datadir, "dataset1.csv")) |> 
    dplyr::select(trialCount, block, acc, RT, stimCongruency, switchType, subject) |> 
    dplyr::mutate(RT = RT/1000,
                  stimCongruency = ifelse(stimCongruency == "i", 1, 0),
                  switchType = ifelse(switchType == "s", 1, 0),
                  subject = match(subject, unique(subject))) |> 
    dplyr::filter(RT < 1.5 & RT > 0.3) |> as.data.table()
  df[,trial := 1:.N, by=.(subject, block)]
  
  return(df)
}

read_dataset_2 <- function(datadir) {
  df <- read.csv(paste0(datadir, "dataset2_replication.csv")) |> 
    dplyr::select(trialCount, block, acc, RT, congruency, taskSequence, subject) |> 
    dplyr::mutate(RT = RT/1000,
                  stimCongruency = ifelse(congruency == "i", 1, 0),
                  switchType = ifelse(taskSequence == "s", 1, 0),
                  subject = match(subject, unique(subject))) |> 
    dplyr::filter(RT < 1.5 & RT > 0.3) |> as.data.table()
  df[,trial := 1:.N, by=.(subject, block)]
  
  return(df)
}

read_dataset_2_forplots <- function(datadir) {
  df <- read_csv(paste0(datadir, "dataset2_replication.csv"), col_types = cols()) |> 
    mutate(RT = RT/1000,
           switchProp = ifelse(blockType == "B" | blockType == "D", "75%", "25%"),
           incProp = ifelse(blockType == "A" | blockType == "B", "75%", "25%")) |> 
    filter(RT < 1.5 & RT > 0.3)
  return(df)
}

read_dataset_1_forplots <- function(datadir) {
  df <- read_csv(paste0(datadir, "dataset1.csv"), col_types = cols()) |> 
    rename(congruency = stimCongruency,
           taskSequence = switchType) |> 
    mutate(RT = RT/1000,
           subject = paste0("sub", match(subject, unique(subject))),
           switchProp = ifelse(blockType == "B" | blockType == "D", "75%", "25%"),
           incProp = ifelse(blockType == "A" | blockType == "B", "75%", "25%")) |> 
    filter(RT < 1.5 & RT > 0.3) |> 
    select(trialCount, block, acc, RT, congruency, taskSequence, switchProp, incProp, subject)
  return(df)
}

get_stan_data <- function(df) {
  stan_data <- list(isInc=df$stimCongruency, isSwitch=df$switchType, RT=df$RT, acc=df$acc,
                    N=nrow(df), M=uniqueN(df$subject), K=3, RTmin=df[,min(RT), by=subject]$V1,
                    trial=df$trial, S=df$subject)
  return(stan_data)
}


fit_loader <- function(complex = T, two_dims = T, replication = F) {
  if (replication) {
    if (complex & two_dims) {
      fitname <- load("fit_2d_full_rep.Rdata")
    } else if (!complex & two_dims) {
      fitname <- load("fit_2d_narrow_rep.Rdata")
    } else if (complex & !two_dims) {
      fitname <- load("fit_1d_rep.Rdata")
    } else if (!complex & !two_dims) {
      fitname <- load("fit_1d_simple_rep.Rdata")
    }
  } else if (!replication) {
    if (complex & two_dims) {
      fitname <- load("fit_2d_full.Rdata")
    } else if (!complex & two_dims) {
      fitname <- load("fit_2d_narrow.Rdata")
    } else if (complex & !two_dims) {
      fitname <- load("fit_1d.Rdata")
    } else if (!complex & !two_dims) {
      fitname <- load("fit_1d_simple.Rdata")
    }
  }
  return(get(fitname))
}

get_rr_pp <- function(fit) spread_draws(fit, RR_pp[t]) |> as.data.table()

get_real_coefs <- function(df, rename_list, include_in_plot_list) {
  
  if (is_empty(include_in_plot_list)) include_in_plot_list <- names(rename_list)
  
  df_means <- df |> 
    group_by(subject, congruency, taskSequence, switchProp, incProp) |> 
    summarise(RT = mean(acc/RT))
  
  # run a linear model on df_means to get the `true` beta effects
  real_lm <- lm(RT ~ congruency * taskSequence * switchProp * incProp, data = df_means)
  
  # get the coefficients
  real_effects <- tibble(as.data.frame(t(coef(real_lm)))) |> 
    rename(!!!rename_list) |> 
    pivot_longer(cols = everything(),
                 names_to = "effect",
                 values_to = ".value") 
  
  # subselect for this plot
  real_plot_effects <- real_effects |> 
    filter(effect %in% include_in_plot_list) |> 
    mutate(model = "Empirical Data")
  
  #fix order
  real_plot_effects$effect = factor(real_plot_effects$effect, levels = rev(include_in_plot_list))
  
  return(real_plot_effects)
}
