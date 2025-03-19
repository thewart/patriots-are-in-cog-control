make_cell_table <- function(posterior_2d, posterior_1d, df_means) {
  posterior_table <- posterior_2d %>% 
    left_join(posterior_1d, by=setdiff(names(posterior_2d), c("RT", ".lower", ".upper")), suffix=c(".2D", ".1D")) %>% 
    left_join(df_means, by=c("congruency", "taskSequence", "switchProp", "incProp")) %>% 
    mutate(.contains.2D = ifelse(mean_RT > .lower.2D & mean_RT < .upper.2D, 1, 0),
           .contains.1D = ifelse(mean_RT > .lower.1D & mean_RT < .upper.1D, 1, 0)) %>% 
    select(switchProp, incProp, congruency, taskSequence, mean_RT, 
           RT.2D, .lower.2D, .upper.2D, .contains.2D,
           RT.1D, .lower.1D, .upper.1D, .contains.1D) %>%
    arrange(switchProp, incProp) 
  
  # mutate posterior_table to add conditional coloring latex text (with cell_spec)
  posterior_table <- posterior_table %>% 
    mutate(across(c(switchProp, incProp), ~ ifelse(.x == "25%", "25\\%", "75\\%"))) %>% 
    mutate(across(c(".contains.2D", ".contains.1D"), ~ cell_spec(.x, 
                                                                 bold = ifelse(.x == 1, TRUE, FALSE),
                                                                 background = ifelse(.x == 1, "#5ec962", "white"),
                                                                 "latex"))) 
  
  # convert to data.frame, change column names, and print latex
  kbl_dat <- data.frame(posterior_table)
  row.names(kbl_dat) <- NULL
  colnames(kbl_dat) <- c("switchProp", "incProp", "congruency", "taskSequence", "mean", 
                         "median", ".lower", ".upper", "*",
                         "median", ".lower", ".upper", "*")
  
  latex_text <- kbl_dat %>% 
    kbl(digits=3, "latex", escape=FALSE,
        align = "ccccccccccccc", booktabs = T) %>% 
    add_header_above(c("Conditions" = 4, "Real Data" = 1, "2D Model" = 4, "1D Model" = 4),
                     bold=T, font_size=12) %>%  
    collapse_rows(columns = 1:4, latex_hline="none")
  
  # fix some latex manually
  change_latex <- function(x, to_remove, replace_with){
    paste(str_replace_all(x, to_remove, replace_with), collapse = "\n")
  }
  
  
  final_latex <- change_latex(latex_text, to_remove = "\\\\cmidrule\\(l\\{3pt\\}r\\{3pt\\}\\)\\{1-1\\}", replace_with = "") %>% 
    change_latex(., to_remove = "5ec962\\}\\{\\\\textbf\\{1\\}", 
                 replace_with = "5ec962\\}\\{\\\\textbf\\{\\\\CheckmarkBold\\}") %>% 
    change_latex(., to_remove = "white\\}\\{0\\}", 
                 replace_with = "white\\}\\{\\{\\\\fontfamily\\{phv\\}\\\\selectfont\\\\textbf\\{X\\}\\}\\}")
  return(final_latex)
}

make_coeff_table <- function(posterior_1d, posterior_2d, real_plot_effects) {
  posterior_table <- rbind(posterior_2d, posterior_1d) %>% 
    mutate(model = ifelse(model == "2D Model", "2D", "1D")) %>% 
    pivot_wider(names_from = model, values_from = c(.value, .lower, .upper), names_sep = ".") %>% 
    left_join(real_plot_effects, by=c("effect")) %>% 
    mutate(.contains.2D = ifelse(.value > .lower.2D & .value  < .upper.2D, 1, 0),
           .contains.1D = ifelse(.value > .lower.1D & .value  < .upper.1D, 1, 0)) %>% 
    select(effect, .value, 
           .value.2D, .lower.2D, .upper.2D, .contains.2D,
           .value.1D, .lower.1D, .upper.1D, .contains.1D) %>% 
    arrange(match(effect, include_in_plot_list))
  
  # mutate posterior_table to add conditional coloring latex text (with cell_spec)
  posterior_table <- posterior_table %>% 
    mutate(across(c(".contains.2D", ".contains.1D"), ~ cell_spec(.x, 
                                                                 bold = ifelse(.x == 1, TRUE, FALSE),
                                                                 background = ifelse(.x == 1, "#5ec962", "white"),
                                                                 "latex"))) 
  
  # convert to data.frame, change column names, and print latex
  kbl_dat <- data.frame(posterior_table)
  row.names(kbl_dat) <- NULL
  colnames(kbl_dat) <- c("", "beta", 
                         "median", ".lower", ".upper", "*",
                         "median", ".lower", ".upper", "*")
  
  #create most of the latex table
  latex_text <- kbl_dat %>% 
    kbl(digits=3, "latex", escape=FALSE,
        align = "lccccccccc", booktabs = T) %>% 
    add_header_above(c("Effect" = 1, "Real Data" = 1, "2D Model" = 4, "1D Model" = 4),
                     bold=T, font_size=12, line=T) %>% 
    collapse_rows(columns = 1:4, latex_hline="none")
  
  # fix some latex manually
  change_latex <- function(x, to_remove, replace_with){
    paste(str_replace_all(x, to_remove, replace_with), collapse = "\n")
  }
  
  
  final_latex <- change_latex(latex_text, to_remove = "\\\\cmidrule\\(l\\{3pt\\}r\\{3pt\\}\\)\\{1-1\\}", replace_with = "") %>% 
    change_latex(., to_remove = "5ec962\\}\\{\\\\textbf\\{1\\}", 
                 replace_with = "5ec962\\}\\{\\\\textbf\\{\\\\CheckmarkBold\\}") %>% 
    change_latex(., to_remove = "white\\}\\{0\\}", 
                 replace_with = "white\\}\\{\\{\\\\fontfamily\\{phv\\}\\\\selectfont\\\\textbf\\{X\\}\\}\\}")
  
  # print out latex with cat to copy paste into overleaf
  return(final_latex)
}
