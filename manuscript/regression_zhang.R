load(url("http://www.umich.edu/~zhanglab/download/Jianrong_2015a/allData.RData"))
library(xtable)
df <- data.frame(
  type = character(),
  species = character(),
  chi = numeric(),
  p = numeric(),
  r2 = numeric(), 
  stringsAsFactors = FALSE 
)
for(type_sp in unique(df.all$type)) {
  df_type <- df.all[df.all$type == type_sp,]
  for(taxon in unique(df_type$species)) {
    print(typeof(taxon))
    f <- df_type$species == taxon;
    e <- df_type$log_expression[f];
    r <- df_type$protein_sequence_pecentage_difference[f];
    # plot(e, r,  main = taxon)
    l <- summary(lm(r~e))
    df[nrow(df) + 1,] = c(type_sp, taxon, l$coefficients[1, 1], l$coefficients[1, 4], l$r.squared)
  }
}
df[] <- lapply(df, function(x) if (anyNA(y <- as.numeric(x))) x else y)
print(xtable(df, digits = 3), include.rownames=FALSE)
print(xtable(df, digits = 3), file = "table.tex", include.rownames=FALSE)
