ll_1d_simple <- extract(fit_1d_simple, "log_lik")[[1]]
ll_1d <- extract(fit_1d, "log_lik")[[1]]
ll_2d_narrow <- extract(fit_2d_narrow, "log_lik")[[1]]
ll_2d_full <- extract(fit_2d_full, "log_lik")[[1]]

gc()
loo_1d <- loo(ll_1d)
gc()
loo_1d_simple <- loo(ll_1d_simple)
gc()
loo_2d_narrow <- loo(ll_2d_narrow)
gc()
loo_2d_full <- loo(ll_2d_full)
gc()

loo::loo_compare(loo_1d, loo_2d_narrow)
