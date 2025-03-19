source("helper_funcs.R")
fit_1d <- fit_loader(two_dims = F, replication = F)

llrhat <- summary(fit_1d, "log_lik")[[1]][,10]
grhat <- summary(fit_1d, "beta")[[1]][,10]
p1 <- rbind(data.table(llrhat, sub=df$subject)[, max(llrhat), by=sub][, type:="Trial~logliklihood"],
      data.table(grhat, sub=str_extract(names(grhat), "\\d+") |> as.numeric())[, max(grhat), by=sub][, type:="psi"]) |>
  ggplot(aes(x=sub, y=V1)) + geom_bar(stat="identity") + coord_cartesian(ylim = c(1, NA)) + theme_classic() + ylab("Maximum Rhat") + xlab("Subject #") + 
  facet_wrap(vars(type), ncol=1, labeller = labeller(type=label_parsed), scales="free")
  
p2 <- stan_dens(fit_1d, c("gamma[28,1]", "gamma[28,2]", "gamma[28,3]"), separate_chains = T)
levels(p2$data$parameter) <- c("psi[s]", "psi[i]", "psi[s %*% i]")
p2 <- p2 + theme_classic() + facet_wrap(vars(parameter), scales = "free", labeller = labeller(parameter=label_parsed)) + ylab("Posterior density") + ggtitle("Subject 28")

llm <- summary(fit_1d, "log_lik")[[2]][df[, which(subject==28)], 1, ]
llm <- data.table(llm-rowMeans(llm), t=1:nrow(llm)) |> melt(id.vars="t")
setnames(llm, "variable", "chain")
levels(llm$chain) <- as.character(1:4)
p3 <- ggplot(llm, aes(y=value, x=t, color=chain)) + geom_point(position = position_dodge(width=1)) + ylab("Centered log-liklihood") + xlab("Trial") + 
  scale_color_manual(values=rstan_options("rstan_chain_cols")[1:4]) + theme_classic()

cowplot::plot_grid(p1, p2, p3, ncol=1, labels = "auto")
