source("helper_funcs.R")
df <- read_dataset_1(datadir)
fit_1d_simple <- fit_loader(complex = F, two_dims = F)
fit_1d <- fit_loader(complex = T, two_dims = F)
fit_2d_full <- fit_loader()

lr <- plogis(rstan::extract(fit_2d_full, "alpha_0_mu")[[1]] |> mean())
omega <- rstan::extract(fit_1d_simple, "omega")[[1]] |> mean()
gamma <- (rstan::extract(fit_1d, "gamma")[[1]] |> apply(c(2,3), mean))[2,]

delta_rule <- function(lr, x) {
  n <- length(x)
  p <- numeric(n)
  p[1] <- 0.5
  
  for (i in 2:n) {
    err <- x[i-1] - p[i-1]
    p[i] <- p[i-1] + lr * err
  }
  return(p)
}

learnedprops <- df[subject==2, .(t=1:.N, incProp=delta_rule(lr, stimCongruency), switchProp=delta_rule(lr, switchType)), by=block]
learnedprops[, phi1 := omega*incProp + (1-omega)*(1-switchProp)]
learnedprops[, phi2 := -(gamma[1]*(switchProp-0.5)/.5 + gamma[2]*(incProp-0.5)/.5 + gamma[3]*(switchProp-0.5)*(incProp-0.5)/.5) + 0.5]
learnedprops <- melt(learnedprops, id.vars = 1:2)
learnedprops[, type:=ifelse(str_detect(variable, "phi"), "Generalized 1D", "2D model")]
learnedprops[variable=="phi1", type:="1D forced-tradeoff"]
learnedprops[variable=="phi1", variable:="phi2"]
learnedprops[, block := ordered(block, labels=c("25% Switch / 25% Incongruent", "75% Switch / 25% Incongruent", "75% Switch / 75% Incongruent", "25% Switch / 75% Incongruent"))]
learnedprops[, type := relevel(factor(type), ref="2D model")]
ggplot(learnedprops, aes(y=value, color=variable, x=t)) + geom_line() + facet_grid(type ~ block) + scale_y_continuous("Control state (a.u.)", n.breaks = 4) +  
  guides(colour = guide_legend(position = "bottom")) + xlab("Trial in block") +
  scale_color_discrete(NULL,  labels=c("Stability level",
                                      "Flexibility level", 
                                      "Stability-flexibility level"))
