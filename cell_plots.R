library(tidyverse)
library(plotrix)
library(cowplot)
library(patchwork)
library(bayesplot)
library(tidybayes)
library(kableExtra)
source("helper_funcs.R")
source("plotters.R")

replication <- T

if (replication) {
  df <- read_csv(paste0(datadir, "dataset2_replication.csv"), col_types = cols()) %>% 
    mutate(RT = RT/1000,
           switchProp = ifelse(blockType == "B" | blockType == "D", "75%", "25%"),
           incProp = ifelse(blockType == "A" | blockType == "B", "75%", "25%"),
           subject = match(subject, unique(subject))) %>% 
    filter(RT < 1.5 & RT > 0.3)
    
} else {
  df <- read_csv(paste0(datadir, "dataset1.csv"), col_types = cols()) %>% 
    rename(congruency = stimCongruency,
           taskSequence = switchType) %>% 
    mutate(RT = RT/1000,
           subject = paste0("sub", match(subject, unique(subject))),
           switchProp = ifelse(blockType == "B" | blockType == "D", "75%", "25%"),
           incProp = ifelse(blockType == "A" | blockType == "B", "75%", "25%")) %>% 
    filter(RT < 1.5 & RT > 0.3) %>% 
    select(trialCount, block, acc, RT, congruency, taskSequence, switchProp, incProp, subject)
}

df_means <- df %>% 
  group_by(subject, congruency, taskSequence, switchProp, incProp) %>% 
  summarise(sub_RT = mean(acc/RT)) %>% 
  group_by(congruency, taskSequence, switchProp, incProp) %>% 
  summarise(mean_RT = mean(sub_RT), sem_RT = 2*std.error(sub_RT))

df <- as.data.table(df)

d1draws <- tidybayes::spread_draws(fit_1d_simple, RR_pp[t]) |> as.data.table()
posterior_1d <- cellwise_posterior(d1draws, df)

d2draws <- tidybayes::spread_draws(fit_2d_full, RR_pp[t]) |> as.data.table()
posterior_2d <- cellwise_posterior(d2draws, df)

df_means <- df_means %>%
  mutate(nudge_direction = ifelse(taskSequence == "s", 0.2, -0.2))

plot_2d <- cell_plot(posterior_2d) + ylab("Correct Response Rate") + 
  theme(legend.position = "bottom",
        strip.text.y = element_blank()) + ggtitle("2D Model (Restricted)")
plot_1d <- cell_plot(posterior_1d) +
  theme(legend.position="none",
        axis.title.y = element_blank(),
        axis.ticks.y = element_blank(),
        axis.text.y = element_blank()) + ggtitle("1D Model (Generalized)")

make_cell_table(posterior_2d, posterior_1d, df_means) |> cat(sep="\n")
