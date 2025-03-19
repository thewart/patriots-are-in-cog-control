df[, c("highSwitch", "highInc") := .(round(mean(switchType)), round(mean(stimCongruency))), by=.(block, subject)]
foo <- tidybayes::spread_draws(fit_1d, RR_pp[t]) |> as.data.table()
foo <- foo[df[, .(t=1:.N, subject, stimCongruency, switchType, highSwitch, highInc)], on="t"]

ppout <- foo[, .(RR_pp=mean(RR_pp)), by=.(.chain, .iteration, subject, stimCongruency, switchType, highSwitch, highInc)][
  , .(RR_pp=mean(RR_pp)), by=.(.chain, .iteration, stimCongruency, switchType, highSwitch, highInc)]



ppout[, dcast(.SD, .iteration ~ .chain, value.var = "RR_pp")[, -1, with=F] |> as.matrix() |> Rhat(),
      by=paste0(stimCongruency, switchType, highSwitch, highInc)]

ppout <- rbind(ppout[, .(mu=mean(RR_pp), lb=quantile(RR_pp, .025), ub=quantile(RR_pp, .975), type="1D model"), by=.(stimCongruency, switchType, highSwitch, highInc)],
               df[, mean(acc/RT), by=.(subject, stimCongruency, switchType, highSwitch, highInc)][
                 , .(mu=mean(V1), lb=mean(V1), ub=mean(V1), type="Empirical"), by=.(stimCongruency, switchType, highSwitch, highInc)])

# ggplot(ppout, aes(y=mu, ymin=lb, ymax=ub, color=type, shape=factor(switchType), x=factor(stimCongruency))) + 
#   geom_pointrange(position = position_dodge(.25)) + facet_grid(highSwitch ~ highInc, labeller = label_both)

juh <- foo[, lm(RR_pp ~ 1 + stimCongruency*switchType)$coef %>% list(variable=names(.), value=.), by=.(subject, .draw, highSwitch, highInc)
           ][, mean(value), by=.(.draw, variable, highSwitch, highInc)
             ][, .(estimate=mean(V1), lb=quantile(V1, .025), ub=quantile(V1, .975)), by=.(variable, highSwitch, highInc)]

ppout <- rbind(cbind(juh, type= "1D model"),
    df[, lm(acc/RT ~ 1 + stimCongruency*switchType)$coef %>% list(variable=names(.), value=.), by=.(subject, highSwitch, highInc)
    ][, .(estimate=mean(value), lb=mean(value)-2*sd(value)/sqrt(.N), ub=mean(value)+2*sd(value)/sqrt(.N)), by=.(variable, highSwitch, highInc)] |> cbind(type="Data"))

foo <- tidybayes::spread_draws(fit_2d_narrow, RR_pp[t]) |> as.data.table()
foo <- foo[df[, .(t=1:.N, subject, stimCongruency, switchType, highSwitch, highInc)], on="t"]
foo <- foo[!(subject %in% badsubj)]

juh2d <- foo[, lm(RR_pp ~ 1 + stimCongruency*switchType)$coef %>% list(variable=names(.), value=.), by=.(subject, .draw, highSwitch, highInc)
           ][, mean(value), by=.(.draw, variable, highSwitch, highInc)
             ][, .(estimate=mean(V1), lb=quantile(V1, .025), ub=quantile(V1, .975)), by=.(variable, highSwitch, highInc)]
ppout <- rbind(ppout, cbind(juh2d, type="2D model"))


foo <- tidybayes::spread_draws(fit_2d_full, RR_pp[t]) |> as.data.table()
foo <- foo[df[, .(t=1:.N, subject, stimCongruency, switchType, highSwitch, highInc)], on="t"]
foo <- foo[!(subject %in% badsubj)]

juh2dfull <- foo[, lm(RR_pp ~ 1 + stimCongruency*switchType)$coef %>% list(variable=names(.), value=.), by=.(subject, .draw, highSwitch, highInc)
           ][, mean(value), by=.(.draw, variable, highSwitch, highInc)
             ][, .(estimate=mean(V1), lb=quantile(V1, .025), ub=quantile(V1, .975)), by=.(variable, highSwitch, highInc)]
ppout <- rbind(ppout, cbind(juh2dfull, type="2D full model"))


ppout[, `Switch proportion` := factor(ppout$highSwitch, labels=c("Low", "High"))]
ppout[, `Incongruent proportion` := factor(ppout$highInc, labels=c("Low", "High"))]
ppout[, variable := factor(variable, levels=unique(variable), labels=c("Baseline", "Congruency effect", "Switch cost", "Switch x Congruency")),]
ggplot(ppout, aes(y=estimate, ymin=lb, ymax=ub, shape=`Switch proportion`, color=`Switch proportion`, x=`Incongruent proportion`)) + 
  geom_pointrange(position = position_dodge(1)) + facet_grid(variable ~ type, scales = "free") + theme_light()


##### acc vs RT
juh <- spread_draws(fit_hier, beta_mu[i, ..]) |> data.table()
juh <- juh[, c(1, 5:6)][, c(lapply(.SD, mean), lapply(.SD, quantile, probs=.025), lapply(.SD, quantile, probs=.975)), by=i]
setnames(juh, c("i", "beta_RT", "beta_acc", "lb_RT", "lb_acc", "ub_RT", "ub_acc") )
ggplot(juh, aes(y=beta_RT, ymin=lb_RT, ymax=ub_RT, x=beta_acc, xmin=lb_acc, xmax=ub_acc)) + geom_point() + 
  geom_errorbar(width=0) + geom_errorbarh(height=0) + geom_hline(yintercept = 0) + geom_vline(xintercept = 0) +
  theme_light() + xlab("Accuracy effect") + ylab("RT effect")


##### 1D pltz
foo <- tidybayes::spread_draws(fit_1d, RT_pp[t], acc_pp[t], controlLevel[t]) |> as.data.table()
uniddat <- foo[, .(RRpp=mean(acc_pp/RT_pp), controlLevel=mean(controlLevel)), by=t
           ][df[, .(t=1:.N, subject, stimCongruency, switchType)], on="t"
             ][, lm(RRpp ~ 1 + stimCongruency*switchType)$coef %>% list(variable=names(.), value=.), by=.(subject, cut_number(controlLevel, 5, labels=F))]
ggplot(juh, aes(x=`cut_number`, y=value, group=paste0(variable, subject))) + geom_line(color="grey", size=0.2) + 
  geom_line(data=juh[,.(subject=-1, value=mean(value, na.rm=T)), by=.(variable, cut_number)]) + 
  facet_wrap(vars(variable), scales="free") + theme_light()


foo2 <- tidybayes::spread_draws(fit_hier, RT_pp[t], acc_pp[t], incProp[t], switchProp[t]) |> as.data.table()
juh <- foo2[, .(RRpp=mean(acc_pp/RT_pp), incProp=mean(incProp), switchProp=mean(switchProp)), by=t
           ][df[, .(t=1:.N, subject, stimCongruency, switchType)], on="t"]

incdat <- juh[, lm(RRpp ~ 1 + stimCongruency*switchType)$coef %>% list(variable=names(.), value=.), by=.(subject, cut_number(incProp, 5, labels=F))]
switchdat <- juh[, lm(RRpp ~ 1 + stimCongruency*switchType)$coef %>% list(variable=names(.), value=.), by=.(subject, cut_number(switchProp, 5, labels=F))]

dimdat <- rbind(cbind(uniddat, type="1D: Control"),
                cbind(incdat, type="2D: Inc. proportion"),
                cbind(switchdat, type="2D: Switch proportion"))
dimdat[, variable := factor(variable, levels=unique(variable), labels=c("Baseline", "Congruency effect", "Switch cost", "Switch x Congruency")),]

ggplot(dimdat, aes(x=`cut_number`, y=value, group=paste0(variable, subject))) + geom_smooth(method="lm", color="grey", size=0.2, se=F) +
  geom_line(data=dimdat[,.(subject=-1, value=mean(value, na.rm=T)), by=.(variable, cut_number, type)]) + 
  facet_grid(variable ~ type, scales="free") + theme_light() + ylab("Reward rate") + xlab("Quintile") + scale_x_reverse()
