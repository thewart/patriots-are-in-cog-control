library(plotrix)
library(patchwork)
library(bayesplot)
library(kableExtra)
library(cowplot)
source("helper_funcs.R")
source("plot_and_table_funcs.R")
replication <- F
complex_1d <- F
complex_2d <- T

# Specify contents of coeff plots/tables ####
rename_list = c(
  "Intercept" = "(Intercept)",
  "Congruency" = "congruencyi",
  "TaskSequence" = "taskSequences",
  "SwitchProp" = "switchProp75%",
  "IncProp" = "incProp75%",
  "Congruency:TaskSequence" = "congruencyi:taskSequences",
  "Congruency:SwitchProp" = "congruencyi:switchProp75%",
  "TaskSequence:SwitchProp" = "taskSequences:switchProp75%",
  "Congruency:IncProp" = "congruencyi:incProp75%",
  "TaskSequence:IncProp" = "taskSequences:incProp75%",
  "SwitchProp:IncProp" = "switchProp75%:incProp75%",
  "Congruency:TaskSequence:SwitchProp" = "congruencyi:taskSequences:switchProp75%",
  "Congruency:TaskSequence:IncProp" = "congruencyi:taskSequences:incProp75%",
  "Congruency:SwitchProp:IncProp" = "congruencyi:switchProp75%:incProp75%",
  "TaskSequence:SwitchProp:IncProp" = "taskSequences:switchProp75%:incProp75%",
  "Congruency:TaskSequence:SwitchProp:IncProp" = "congruencyi:taskSequences:switchProp75%:incProp75%"
)

include_in_plot_list = c(
  "TaskSequence",
  "Congruency",
  "TaskSequence:SwitchProp",
  "TaskSequence:IncProp",
  "Congruency:SwitchProp",
  "Congruency:IncProp"
)

# Load data and fits ####

if(replication) {
  df <- read_dataset_2_forplots(datadir)
} else {
  df <- read_dataset_1_forplots(datadir)
}

d1draws <- fit_loader(complex_1d, two_dims = F, replication) |> get_rr_pp()
d2draws <- fit_loader(complex_2d, two_dims = T, replication) |> get_rr_pp()

# Coefficient plots/tables ####
real_plot_effects <- get_real_coefs(df, rename_list, include_in_plot_list)
d1coeff_post <- coef_posterior(d1draws, as.data.table(df), rename_list, include_in_plot_list, "1D Model")
d2coeff_post <- coef_posterior(d2draws, as.data.table(df), rename_list, include_in_plot_list, "2D Model")

coef_plt <- effects_plot(d1coeff_post, d2coeff_post, real_plot_effects)
coef_table <- make_coeff_table(d1coeff_post, d2coeff_post, real_plot_effects) |> cat(sep="\n")

# cells ####
df_means <- df |>
  group_by(subject, congruency, taskSequence, switchProp, incProp) |>
  summarise(sub_RT = mean(acc/RT)) |>
  group_by(congruency, taskSequence, switchProp, incProp) |>
  summarise(mean_RT = mean(sub_RT), sem_RT = 2*std.error(sub_RT))

d1cell_post <- cellwise_posterior(d1draws, as.data.table(df))
d2cell_post <- cellwise_posterior(d2draws, as.data.table(df))

cell_1d_plt <- cell_plot(d1cell_post, df_means)
cell_2d_plt <- cell_plot(d2cell_post, df_means)

cell_table <- make_cell_table(d2cell_post, d1cell_post, df_means) |> cat(sep="\n")

# Put together subfigures ####
a_plt <- cell_2d_plt + ggtitle("2D Model") + 
  theme(legend.position = "bottom", strip.text.y = element_blank())
  
b_plt <- cell_1d_plt + ggtitle("1D Model") + 
  theme(legend.position="none",axis.title.y = element_blank(),
        axis.ticks.y = element_blank(), axis.text.y = element_blank())

bigfig <- plot_grid(a_plt+b_plt, coef_plt, nrow=2, rel_heights = c(1.3,1), labels="auto")

