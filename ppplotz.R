df[, c("highSwitch", "highInc") := .(round(mean(switchType)), round(mean(stimCongruency))), by=.(block, subject)]
foo <- spread_draws(fit_1d, RT_pp[t], acc_pp[t]) |> as.data.table()
foo <- foo[, .(.draw, t, RRpp=acc_pp/RT_pp)]
foo <- foo[df[, .(t=1:.N, subject, stimCongruency, switchType, highSwitch, highInc)], on="t"]
ppout <- foo[, .(RRpp=mean(RRpp)), by=.(.draw, subject, stimCongruency, switchType, highSwitch, highInc)][
  , .(RRpp=mean(RRpp)), by=.(.draw, stimCongruency, switchType, highSwitch, highInc)][
    , .(mu=mean(RRpp), lb=quantile(RRpp, .025), ub=quantile(RRpp, .975), type="pp"), by=.(stimCongruency, switchType, highSwitch, highInc)]
ppout <- rbind(ppout, 
               df[, mean(acc/RT), by=.(subject, stimCongruency, switchType, highSwitch, highInc)][
                 , .(mu=mean(V1), lb=mean(V1), ub=mean(V1), type="obs"), by=.(stimCongruency, switchType, highSwitch, highInc)])

ggplot(ppout, aes(y=mu, ymin=lb, ymax=ub, color=type, shape=factor(switchType), x=factor(stimCongruency))) + 
  geom_pointrange(position = position_dodge(.25)) + facet_grid(highSwitch ~ highInc, labeller = label_both)
