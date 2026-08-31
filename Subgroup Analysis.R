# ==============================================================================
# GBM-Specific SHAP Dependence Analysis: Evaluating Enzyme-Severity Dissociation
# ==============================================================================
library(ggplot2)

cat("\n=================================================================\n")
cat("==== Stratified SHAP Dependence Analysis (Supplementary Fig) ====\n")
cat("=================================================================\n")

# 1. Data alignment and completeness check
# Ensure row indices match between traindata and SHAP matrices to prevent ggplot errors
vars_needed <- c("ALT", "Cirrhosis")
complete_idx <- complete.cases(traindata[, vars_needed])

analysis_df <- traindata[complete_idx, ]
analysis_df$Cirrhosis <- as.numeric(as.character(analysis_df$Cirrhosis))

# Extract and synchronize SHAP data
shap_ALT <- shap_value$X[complete_idx, "ALT"]
shap_val <- shap_value$S[complete_idx, "ALT"]
filtered_cirrhosis <- analysis_df$Cirrhosis

cat(sprintf("[Info] Valid sample size for SHAP analysis: N = %d\n", length(shap_ALT)))

# 2. Construct data frame for plotting
plot_shap_df <- data.frame(
  ALT = shap_ALT,
  SHAP = shap_val,
  Cirrhosis_Label = factor(filtered_cirrhosis, levels = c(0, 1), labels = c("Non-Cirrhotic", "Cirrhotic"))
)

# 3. Plot stratified SHAP dependence
p_shap <- ggplot(plot_shap_df, aes(x = ALT, y = SHAP, color = Cirrhosis_Label)) +
  geom_point(alpha = 0.8, size = 2.2, shape = 1, stroke = 0.8) +
  geom_smooth(method = "loess", se = TRUE, linewidth = 0.6) + 
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50", linewidth = 0.5) +
  scale_color_manual(values = c("Non-Cirrhotic" = "#2c7bb6", "Cirrhotic" = "#d7191c"), na.translate = FALSE) + 
  theme_bw(base_size = 14) +                  
  labs(
    title = "Supplementary Fig 4B: Stratified SHAP Dependence", 
    subtitle = "Evaluating the Enzyme-Severity Dissociation by Cirrhosis Status",
    x = "ALT Level (U/L)",
    y = "SHAP Value (Impact on Treatment Failure Risk)",
    color = "Clinical Status"
  ) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
    plot.subtitle = element_text(hjust = 0.5, face = "italic", size = 11),
    axis.title = element_text(face = "bold"),
    legend.position = "right",
    legend.title = element_text(face = "bold"),
    panel.grid.major = element_line(color = "gray85", linewidth = 0.3),
    panel.grid.minor = element_blank() 
  )

# 4. Print and save plot
print(p_shap)
ggsave("Supplementary_Fig_4B_Stratified_SHAP.pdf", plot = p_shap, width = 7, height = 5, dpi = 300)
cat("[Success] Stratified SHAP dependence plot saved.\n")

# ==============================================================================
# Subgroup AUC Analysis (Overlap, Cirrhosis, Liver Failure)
# ==============================================================================
library(pROC)

# 1. Define function to generate subgroup AUC table with DeLong P-values
get_subgroup_auc_table <- function(data, subgroup_col, group_name, labels_dict) {
  levels_val <- sort(unique(na.omit(data[[subgroup_col]])))
  res_list <- list()
  roc_list <- list() # Store ROC objects for DeLong testing
  
  for (lvl in levels_val) {
    sub_data <- data[data[[subgroup_col]] == lvl, ]
    n_total <- nrow(sub_data)
    
    if (length(unique(sub_data$outcome)) < 2) {
      auc_str <- "Not calculable"
    } else {
      # Calculate ROC and DeLong's 95% CI
      roc_obj <- pROC::roc(sub_data$outcome, sub_data$prob, quiet = TRUE)
      ci_obj <- pROC::ci.auc(roc_obj, method = "delong")
      auc_str <- sprintf("%.3f (%.3f - %.3f)", roc_obj$auc, ci_obj[1], ci_obj[3])
      roc_list[[as.character(lvl)]] <- roc_obj
    }
    
    category_name <- labels_dict[as.character(lvl)]
    res_list[[length(res_list) + 1]] <- data.frame(
      Variable = group_name, 
      Subgroup = category_name, 
      N = n_total,
      `AUC (95% CI)` = auc_str, 
      check.names = FALSE
    )
  }
  
  df_res <- do.call(rbind, res_list)
  
  # Calculate DeLong P-value for subgroups with exactly two categories
  if (length(roc_list) == 2) {
    delong_test <- pROC::roc.test(roc_list[[1]], roc_list[[2]], method = "delong")
    p_val <- delong_test$p.value
    
    # Format P-value: display <0.001 if applicable, else 3 decimal places
    p_val_str <- ifelse(p_val < 0.001, "<0.001", sprintf("%.3f", p_val))
    
    # Place P-value in the first row only for clean formatting
    df_res$`DeLong P-value` <- c(p_val_str, "") 
  } else {
    df_res$`DeLong P-value` <- rep("-", nrow(df_res))
  }
  
  return(df_res)
}

# 2. Execute table generation and merge
dict_overlap <- c("0" = "Classic AIH", "1" = "AIH-PBC Overlap")
dict_cirrhosis <- c("0" = "Non-Cirrhotic", "1" = "Cirrhotic")
dict_failure <- c("0" = "Without Liver Failure", "1" = "With Liver Failure")

# Assuming 'sub_df' is prepared in the preceding environment
tab_overlap <- get_subgroup_auc_table(sub_df, "Overlap", "Target Population", dict_overlap)
tab_cirrhosis <- get_subgroup_auc_table(sub_df, "Cirrhosis", "Cirrhosis Status", dict_cirrhosis)
tab_failure <- get_subgroup_auc_table(sub_df, "LiverFailure", "Liver Failure Status", dict_failure)

# Combine all subgroup tables
supp_table <- rbind(tab_overlap, tab_cirrhosis, tab_failure)

# Optimize formatting: display category name only in the first row
if(nrow(supp_table) == 6) {
  supp_table$Variable[c(2, 4, 6)] <- ""
}

# 3. Export to CSV
file_name <- "Supplementary_Table_Subgroup_AUC_DeLong.csv"
write.csv(supp_table, file_name, row.names = FALSE, fileEncoding = "UTF-8")

cat(sprintf("\n[Success] Subgroup AUC table with DeLong P-values exported as: %s\n", file_name))