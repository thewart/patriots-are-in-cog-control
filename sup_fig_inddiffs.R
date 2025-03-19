source("helper_funcs.R")
library(GGally)
df <- read_dataset_1_forplots(datadir)
d2draws <- fit_loader() |> get_rr_pp()

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

create_arrow_data <- function(df) {
  result <- data.frame()
  
  for (s in unique(df$subject)) {
    # Get the 4 points for this subject
    subj_points <- df %>% filter(subject == s)
    
    if (nrow(subj_points) == 4) {  # Ensure we have all 4 conditions
      # Create all combinations of rows
      for (i in 1:3) {
        for (j in (i+1):4) {
          row_i <- subj_points[i, ]
          row_j <- subj_points[j, ]
          
          # Check if exactly one property is shared (XOR)
          same_switch <- row_i$`Switch Prop` == row_j$`Switch Prop`
          same_incongruency <- row_i$`Incongruency Prop` == row_j$`Incongruency Prop`
          
          if ((same_switch && !same_incongruency) || (!same_switch && same_incongruency)) {
            # These points share exactly one property
            result <- rbind(result, data.frame(
              subject = s,
              x = row_i$TaskSequence,
              y = row_i$Congruency,
              xend = row_j$TaskSequence,
              yend = row_j$Congruency
            ))
          }
        }
      }
    }
  }
  
  return(result)
}


d2draws <- d2draws[as.data.table(df)[, .(t=1:.N, subject, congruency, taskSequence, switchProp, incProp)], on="t"]
juh <- d2draws[, lm(RR_pp ~ 1 + congruency*taskSequence*switchProp*incProp)$coef %>% list(variable=names(.), value=.), by=.(.draw, subject)][
  , mean(value), by = .(variable, subject)] |> dcast(subject ~ variable)

setnames(juh, rename_list, names(rename_list))

juh[, .(Congruency, TaskSequence, `Congruency:IncProp`, `TaskSequence:SwitchProp`)] |> ggpairs()

juh[, .(`Congruency:IncProp`, `TaskSequence:SwitchProp`, 
        `Congruency:SwitchProp:IncProp`, `TaskSequence:SwitchProp:IncProp`)] |> ggpairs()

coneff <- juh[, .(subject, `25/25` = Congruency,
                  `75/25` = Congruency + `Congruency:IncProp`,
                  `25/75` = Congruency + `Congruency:SwitchProp`,
                  `75/75` = Congruency + `Congruency:IncProp` + `Congruency:SwitchProp` + `Congruency:SwitchProp:IncProp`)]

swicst <- juh[, .(subject, `25/25` = TaskSequence,
                  `75/25` = TaskSequence + `TaskSequence:IncProp`,
                  `25/75` = TaskSequence + `TaskSequence:SwitchProp`,
                  `75/75` = TaskSequence + `TaskSequence:IncProp` + `TaskSequence:SwitchProp` + `TaskSequence:SwitchProp:IncProp`)]

conswi <- melt(coneff, value.name = "Congruency")[melt(swicst, value.name = "TaskSequence"), on = .(subject, variable)]
conswi[, c("Incongruency Prop", "Switch Prop") := .(str_split_i(variable, "/", 1) |> str_c("%"),
                                                  str_split_i(variable, "/", 2) |> str_c("%"))]

conswi_plt <- ggplot(conswi, aes(y = Congruency, x = TaskSequence, color = `Switch Prop`, shape = `Incongruency Prop`)) + 
  geom_segment(data = arrow_data, 
               aes(x = x, y = y, xend = xend, yend = yend),
               alpha = 0.15,
               inherit.aes = FALSE) + 
  geom_point(size = 2) + geom_vline(xintercept = 0, alpha = 0.6) + geom_hline(yintercept = 0, alpha = 0.6)



