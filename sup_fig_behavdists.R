source("helper_funcs.R")
library(cowplot)

df <- read_dataset_1_forplots(datadir)
indplts <- df |> group_by(subject, taskSequence, congruency, switchProp, incProp) |> summarise(RT = mean(RT), acc = mean(acc))
plt_1 <- indplts |> ggplot(aes(y=RT, x = taskSequence, fill = congruency)) + geom_boxplot() + facet_grid(incProp ~ switchProp) +
  facet_grid(switchProp ~ incProp, labeller = labeller(incProp = c("25%" = "25% Incongruent", "75%" = "75% Incongruent"),
                                                       switchProp = c("25%" = "25% Switch", "75%" = "75% Switch"))) + 
  scale_fill_manual(name="Congruency", values = c("c"="#AA9355", "i"="#843C0C"), labels = c("c"="Congruent", "i"="Incongruent")) +
  scale_x_discrete(labels=c("r" = "Repeat", "s" = "Switch")) + xlab("Task Sequence") + ylab("Response Time")
  
plt_2 <- indplts |> ggplot(aes(y=acc, x = taskSequence, fill = congruency)) + geom_boxplot() + facet_grid(incProp ~ switchProp) +
  facet_grid(switchProp ~ incProp, labeller = labeller(incProp = c("25%" = "25% Incongruent", "75%" = "75% Incongruent"),
                                                       switchProp = c("25%" = "25% Switch", "75%" = "75% Switch"))) + 
  scale_fill_manual(name="Congruency", values = c("c"="#AA9355", "i"="#843C0C"), labels = c("c"="Congruent", "i"="Incongruent")) +
  scale_x_discrete(labels=c("r" = "Repeat", "s" = "Switch")) + xlab("Task Sequence") + ylab("Accuracy")

plot_grid(plt_1, plt_2, ncol = 1)
