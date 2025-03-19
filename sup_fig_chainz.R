source("helper_funcs.R")
library(ggh4x)
library(cowplot)
fit_1d <- fit_loader(two_dims = F, replication = F)

allthesamps <- rbind(stan_trace(fit_1d, "beta_mu")$data, stan_trace(fit_1d, "a_mu")$data) |> as.data.table()
intinds <- allthesamps[, str_detect(parameter, "\\[[4-7],")]
allthesamps[, btype := fifelse(str_detect(parameter, "1\\]"), "RT", "acc")]
slopes <- allthesamps[intinds]
intercepts <- allthesamps[!intinds]

slopes[, efftype := str_extract(parameter, "\\d") |> case_match(
  "4" ~ "Baseline",
  "5" ~ "Congruency effect",
  "6" ~ "Switch cost",
  "7" ~ "Switch x congruency"
)]

intercepts[, efftype := case_when(
  str_detect(parameter, "^a_mu") ~ "Baseline",
  str_detect(parameter, "1,") ~ "Congruency effect",
  str_detect(parameter, "2,") ~ "Switch cost",
  str_detect(parameter, "3,") ~ "Switch x congruency"
)]

plt_1 <- ggplot(intercepts, aes(y = value, x = iteration, color = chain)) + geom_line() + 
  facet_grid2(efftype ~ btype, scales = "free", independent = "y") + ggtitle(expression("Intercepts ("*M[0]*")"))
plt_2 <- ggplot(slopes, aes(y = value, x = iteration, color = chain)) + geom_line() + 
  facet_grid2(efftype ~ btype, scales = "free", independent = "y") + ggtitle(expression("Slopes ("*M[phi]*")"))

plot_grid(plt_1, plt_2, ncol = 1)
