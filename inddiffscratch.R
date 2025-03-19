subjll_1d <- setDF(gather_draws(fit_1d, log_lik[t]))[
  setDT(df[, .(t=1:N, subject)]), on="t"
  ][, mean(.value), by=.(subject, t)][, sum(V1), by=subject]
subjll_2d <- setDF(gather_draws(fit_2d_full, log_lik[t]))[
  setDT(df[, .(t=1:N, subject)]), on="t"
  ][, mean(.value), by=.(subject, t)][, sum(V1), by=subject]

gamma <- setDT(gather_draws(fit_1d, gamma[s, i]))[, median_qi(.value), by=.(s,i)]
subj <- subjll_2d[(V1-subjll_1d$V1)<1, str_extract(subject, "\\d+")] |> as.numeric()

gamma[s %in%subj]
