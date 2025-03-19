library(tidybayes)
foo <- spread_draws(fit_2d_full, RR_mu[i,j,..]) |> as.data.table()

setnames(foo, 6:9, c("Baseline", "Inc", "Switch", "IncxSwitch"))
foo[, c("ConEff", "SwitchCost") := .(Inc-Baseline, Switch-Baseline)]

# foo[, efftype:= factor(k, labels = c("Baseline", "Inc", "Switch", "IncxSwitch"))]
foo[, incProp:= ordered(i, labels = seq(-1,1,0.25))]
foo[, switchProp:= ordered(j, labels = seq(-1,1,0.25))]
foo <- foo[(switchProp %in% seq(-1,1,.5)) & (incProp %in% seq(-1,1,.5))]
foo <- melt(foo[, .(incProp, switchProp, Baseline, ConEff, SwitchCost)])[
  , .(mu=mean(value), sderr=sd(value)/sqrt(.N)), by=.(incProp, switchProp, variable)] |> 
  dcast(incProp + switchProp ~ variable, value.var = c("mu", "sderr"))

ggplot(foo, aes(y=mu_SwitchCost, x=switchProp, group=incProp, color=incProp)) + geom_point() + geom_line()
ggplot(foo, aes(y=mu_ConEff, x=incProp, group=switchProp, color=switchProp)) + geom_point() + geom_line()
ggplot(juh, aes(y=Baseline, x=switchProp, group=incProp, color=incProp)) + geom_point() + geom_line()
ggplot(juh, aes(y=IncxSwitch-(Baseline + (Inc-Baseline) + (Switch-Baseline)), x=switchProp, group=incProp, color=incProp)) + geom_point() + geom_line()

ggplot(juh, aes(y=Switch-Baseline, x=Inc-Baseline, group=incProp, color=switchProp)) + geom_point() + geom_line()
