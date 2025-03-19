juh <- summary(fit_1d, "gamma")[[1]]
gamma <- data.table(subj=rownames(juh) |> str_extract("\\d+"), thisname=rownames(juh), rhat=juh[,10])
badsubj <- gamma[, max(rhat), by=subj][V1>1.2, subj]
# write(badsubj, "theybad.csv", sep=",")

