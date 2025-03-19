library(tidybayes)
postsims_wide <- getbaselinecomps(fit_2d_full, SDoffset = 0)
postsims_wide[, c("incProp", "switchProp") := .(incProp>0, switchProp>0)]

baseline <- postsims_wide[, lm(Baseline ~ incProp * switchProp)$coef %>% list(variable=names(.), value=.), .draw]
switchTrials <- postsims_wide[, lm(Switch ~ incProp * switchProp)$coef %>% list(variable=names(.), value=.), .draw]
incTrials <- postsims_wide[, lm(Inc ~ incProp * switchProp)$coef %>% list(variable=names(.), value=.), .draw]

baseline[, mean_qi(value), by=variable]
switchTrials[, mean_qi(value), by=variable]
incTrials[, mean_qi(value), by=variable]

(baseline[variable=="switchProp", value] + switchTrials[variable=="switchProp", value]) |> mean_qi()
(baseline[variable=="incProp", value] + incTrials[variable=="incProp", value]) |> mean_qi()
(baseline[variable=="switchPropTRUE", value] - incTrials[variable=="switchPropTRUE", value]) |> mean_qi()

postsims_wide0 <- getbaselinecomps(fit_2d_full, SDoffset = 0)
postsims_wide_p1 <- getbaselinecomps(fit_2d_full, SDoffset = 1)
postsims_wide_m1 <- getbaselinecomps(fit_2d_full, SDoffset = -1)
postsims_wide <- rbind(
  cbind(postsims_wide0, SD=0),
  cbind(postsims_wide_p1, SD=1),
  cbind(postsims_wide_m1, SD=-1)
)
postsims_wide[, c("incProp", "switchProp") := .(incProp>0, switchProp>0)]

postsims_wide[, mean_qi(Baseline, .width = .8), by=.(incProp, switchProp, SD)] |> 
  ggplot(aes(y=y, x=ordered(incProp), group=ordered(switchProp), shape=ordered(switchProp), ymin=ymin, ymax=ymax)) + 
  geom_pointrange(position = position_dodge(0.25)) + geom_line(position = position_dodge(0.25)) + facet_wrap(vars(SD))

baseline <- postsims_wide[, lm(Baseline ~ incProp * switchProp)$coef %>% list(variable=names(.), value=.), by=.(.draw, SD)]
baseline[, mean_qi(value), by=.(variable, SD)]

varplt <- postsims_wide[, mean_qi(Baseline, .width = .8), by=.(incProp, switchProp, SD)] |>
  ggplot(aes(y=y, x=ordered(incProp), group=ordered(switchProp), shape=ordered(switchProp), ymin=ymin, ymax=ymax)) +
  geom_pointrange(position = position_dodge(0.25)) + geom_line(position = position_dodge(0.25)) +
  facet_wrap(vars(SD), labeller = labeller(SD=c("-1" = "-1 SD", "0"="+0 SD", "1"="+1 SD"))) +
  scale_x_discrete("Learned congrueny proportion", labels = c("25%", "75%")) +
  scale_shape_manual("Learned\nswitch proportion", values = c(16, 17), labels=c("25%", "75%")) + ylab("Congruent/repeat trial CRR")

ggsave("~/Desktop/fig5.png", varplt, width = 6.5, height = 3)
