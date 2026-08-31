# ==============================================================================
# 5. Temporal Validation (Test 2): Predictions, Evaluation, and Subgroups
# ==============================================================================

library(naniar)
library(mice) 
library(tableone)
library(ggplot2)
library(lattice)
library(caret)
library(autoReg)
library(dplyr)
library(Matrix)
library(glmnet)
library(Boruta)
library(Hmisc)
library(rms)
library(pROC)
library(rmda)

# ============================================================
# Step 1. Data Loading and Preprocessing
# ============================================================
setwd("/Users/zhoushujie/Desktop")

if (.Platform$OS.type == "windows") {
  windowsFonts(Times = windowsFont("Times New Roman")) 
}
par(family = "Times")

# Load raw validation data
data <- read.csv("validation data.csv", header = TRUE, encoding = "GBK")

# Batch convert categorical variables to factors
data[,1:24] <- lapply(data[,1:24], as.factor) 

# Missing value handling & imputation
data %>%
  gg_miss_var(show_pct = TRUE) + 
  theme_bw(base_size = 16, base_family = "Times") 

missing_ratio <- colMeans(is.na(data))
missing_ratio_df <- data.frame(Variable = names(missing_ratio), Missing_Ratio = missing_ratio)
write.csv(missing_ratio_df, "Missing_Value_Ratio_Test2.csv", row.names = FALSE)

# Multiple imputation (m=5)
data_imputed <- mice(data, seed = 123, print = FALSE, m = 5)
data_imputed2 <- mice::complete(data_imputed, action = 2)
write.csv(data_imputed2, "test.csv", row.names = FALSE)

# Read imputed data
test2_data <- read.csv("test.csv", header = TRUE, encoding = "GBK")
test2_data$outcome <- factor(test2_data$outcome, levels = c(0,1), labels = c('improvement','deteriorate'))
y_test2 <- test2_data$outcome

# Baseline table generation for Test 2
myVars <- colnames(test2_data[, 2:ncol(test2_data)])  
catVars <- colnames(test2_data[, 2:25])         
contVars <- colnames(test2_data[, 26:ncol(test2_data)])  

noncenter <- c()
for (var in contVars) {
  if (is.numeric(test2_data[[var]])) {
    test_result <- shapiro.test(test2_data[[var]])
    if (test_result$p.value < 0.05) {
      noncenter <- c(noncenter, var)
    }
  }
}

table1 <- CreateTableOne(vars = myVars, factorVars = catVars, data = test2_data, strata='outcome', addOverall = TRUE)  
table1 <- print(table1, showAllLevels = TRUE, catDigits = 2, contDigits = 2, pDigits = 3, nonnormal = noncenter)
write.csv(table1, file = "Test2_Baseline_Table.csv") 

# ============================================================
# Step 2. Feature Encoding and Probability Extraction
# ============================================================

# Isolate predictors
Xte2_raw <- test2_data %>% select(-outcome)
Xte2_raw <- Xte2_raw %>% mutate(ANA = as.factor(ANA))

# Apply one-hot encoding using the pre-defined 'dv' rule from the training set to prevent data leakage
X_test2 <- as.data.frame(predict(dv, newdata = Xte2_raw))

# Standardize binary outcome labels
test2_y <- factor(ifelse(y_test2 == 1 | y_test2 == "deteriorate", "fail", "improve"), levels = c("fail", "improve"))

test2_m <- data.frame(
  outcome = test2_y,
  X_test2[, feature_names, drop = FALSE]
)

# Extract predicted probabilities for all models in Test 2
cat("\n[Info] Extracting predicted probabilities for Test 2...\n")
pred_test2 <- data.frame(outcome = test2_m$outcome)

for (mn in names(models)) {
  pred_test2[[mn]] <- get_prob_fail(models[[mn]], mn, test2_m, outcome_col = "outcome")
}

datalist_test2 <- list(Test2 = pred_test2)

# ============================================================
# Step 3. Visualization: Calibration, ROC, CM, & DCA
# ============================================================

# ------------------------------------------------------------
# 3.1 Test 2 Calibration Curves
# ------------------------------------------------------------
cat("\n[Info] Plotting Test 2 Calibration Curves...\n")
for (tt in names(datalist_test2)) {
  newdata <- datalist_test2[[tt]]
  cutpoint <- 5  
  
  formula_cal <- as.formula(paste0("outcome ~ ", paste(colnames(newdata)[-1], collapse = " + ")))
  cal_obj <- caret::calibration(formula_cal, data = newdata, class = "fail", cuts = cutpoint)
  caldata <- as.data.frame(cal_obj$data) %>% tidyr::drop_na()
  
  model_order <- colnames(newdata)[-1]
  caldata$calibModelVar <- factor(caldata$calibModelVar, levels = model_order)
  cm <- get_color_map(model_order)
  
  Calibrat_plot <- ggplot(caldata, aes(x = midpoint, y = Percent, group = calibModelVar, color = calibModelVar)) +
    geom_point(size = 1.6) + geom_line(linewidth = 0.8) +
    geom_abline(slope = 1, intercept = 0, color = "black", linetype = "dashed") +
    scale_color_manual(values = cm, name = "Model", drop = FALSE) +
    labs(title = paste0(tt, " Calibration"), x = "Predicted risk (bin midpoint)", y = "Observed event rate") +
    theme_bw(base_family = "serif") +
    theme(
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
      legend.position = c(0.80, 0.25),
      panel.border = element_rect(color = "black", linewidth = 1)
    )
  
  pdf(paste0(tt, "_Calibration.pdf"), 5.5, 5.5, family = "serif")
  print(Calibrat_plot)
  dev.off()
}

# ------------------------------------------------------------
# 3.2 Test 2 ROC Curves & Metrics Evaluation
# ------------------------------------------------------------
cat("[Info] Plotting Test 2 ROC and calculating metrics...\n")
for (tt in names(datalist_test2)) {
  newdata <- datalist_test2[[tt]]
  ROC_list  <- list()
  ROC_label <- list()
  Evaluation_metrics <- data.frame()
  
  for (mn in colnames(newdata)[-1]) {
    prob <- as.numeric(newdata[[mn]])
    if (all(is.na(prob)) || length(unique(prob[!is.na(prob)])) < 2) next
    
    ROC <- pROC::roc(response = newdata$outcome, predictor = prob, levels = c("improve","fail"), direction = "<", quiet = TRUE)
    AUC <- as.numeric(pROC::auc(ROC))
    CI  <- as.numeric(pROC::ci.auc(ROC))
    bestp <- get_best_threshold(ROC)
    
    predlab <- factor(ifelse(prob > bestp, "fail", "improve"), levels = c("improve","fail"))
    index_table <- caret::confusionMatrix(data = predlab, reference = newdata$outcome, positive = "fail", mode = "everything")
    
    cm_plot <- plot_cm_gg(ref = newdata$outcome, pred = predlab, title = paste0(tt, " - ", mn))
    pdf(paste0(tt, "_", mn, "_CM.pdf"), 5, 5, family = "serif")
    print(cm_plot)
    dev.off()
    
    Evaluation_metrics <- rbind(Evaluation_metrics, data.frame(
      Model = mn, Threshold = bestp, AUC = AUC, CI_low = CI[1], CI_high = CI[3],
      Accuracy = as.numeric(index_table$overall["Accuracy"]),
      Sensitivity = as.numeric(index_table$byClass["Sensitivity"]),
      Specificity = as.numeric(index_table$byClass["Specificity"]),
      Precision = as.numeric(index_table$byClass["Precision"]),
      F1 = as.numeric(index_table$byClass["F1"]), stringsAsFactors = FALSE
    ))
    
    ROC_label[[mn]] <- paste0(mn, " (AUC=", sprintf("%0.3f", AUC), ", 95% CI: ", sprintf("%0.3f", CI[1]), "-", sprintf("%0.3f", CI[3]), ")")
    ROC_list[[mn]] <- ROC
  }
  
  write.csv(Evaluation_metrics, paste0(tt, "_Evaluation_metrics.csv"), row.names = FALSE)
  
  if (length(ROC_list) > 0) {
    roc_models <- names(ROC_list)
    cm <- get_color_map(roc_models)
    ROC_plot <- pROC::ggroc(ROC_list, size = 1.2, legacy.axes = TRUE) + theme_bw() +
      labs(title = paste0(tt, " ROC curve")) +
      theme(
        plot.title = element_text(hjust = 0.5, size = 15, face = "bold"),
        legend.position = c(0.70, 0.25), legend.title = element_blank(),
        panel.border = element_rect(color="black", linewidth = 1)
      ) +
      geom_segment(aes(x = 0, y = 0, xend = 1, yend = 1), colour = "grey", linetype = "dotdash") +
      scale_colour_manual(values = cm, breaks = roc_models, labels = unname(ROC_label[roc_models]))
    
    pdf(paste0(tt, "_ROC.pdf"), 7, 7, family = "serif")
    print(ROC_plot)
    dev.off()
  }
}

# ------------------------------------------------------------
# 3.3 Test 2 Decision Curve Analysis (DCA)
# ------------------------------------------------------------
cat("[Info] Plotting Test 2 DCA...\n")
for (tt in names(datalist_test2)) {
  newdata <- datalist_test2[[tt]]
  dca_data <- newdata
  dca_data$outcome <- ifelse(dca_data$outcome == "fail", 1, 0)
  DCA_list <- list()
  
  for (mn in colnames(dca_data)[-1]) {
    prob <- as.numeric(dca_data[[mn]])
    if (all(is.na(prob)) || length(unique(prob[!is.na(prob)])) < 2) next
    
    set.seed(123)
    DCA_list[[mn]] <- rmda::decision_curve(
      formula = as.formula(paste("outcome ~", mn)), data = dca_data, family = binomial(link = "logit"),
      thresholds = seq(0.05, 0.95, by = 0.01), confidence.intervals = FALSE, study.design = "cohort"
    )
  }
  
  if (length(DCA_list) > 0) {
    dca_obj <- setNames(DCA_list, names(DCA_list))
    dca_models <- names(dca_obj)
    cm <- get_color_map(dca_models)
    col_vec <- unname(cm[dca_models])
    
    pdf(paste0(tt, "_DCA.pdf"), 7, 7, family = "serif")
    rmda::plot_decision_curve(
      dca_obj, curve.names = dca_models, col = col_vec, lty = rep(1, length(dca_models)),
      lwd = 2, confidence.intervals = FALSE, cost.benefit.axis = FALSE, legend.position = "topright"
    )
    title(main = paste0(tt, " Decision Curve Analysis"))
    dev.off()
  }
}
cat("[Success] All Test 2 visualizations generated and saved.\n")


# ============================================================
# Step 4. Subgroup Analysis with DeLong Test (Test 2)
# ============================================================
cat("\n[Info] Constructing Test 2 subgroup analysis dataframe...\n")

test2_y_numeric <- ifelse(test2_data$outcome == "deteriorate", 1, 0)

sub_df_test2 <- data.frame(
  outcome = test2_y_numeric,
  prob = as.numeric(pred_test2$GBM),      
  Overlap = test2_data$PBC,               
  Cirrhosis = test2_data$Cirrhosis,       
  LiverFailure = test2_data$hepatic_failure 
)
cat("[Success] Subgroup dataframe built. Rows:", nrow(sub_df_test2), "\n")

# Function to calculate AUC per subgroup and execute DeLong test
calc_subgroup_auc <- function(data, subgroup_col, group_name, labels_dict) {
  cat(sprintf("\n>>> [ Subgroup Dimension: %s ] <<<\n", group_name))
  levels_val <- sort(unique(na.omit(data[[subgroup_col]])))
  roc_list <- list() 
  
  for (lvl in levels_val) {
    sub_data <- data[data[[subgroup_col]] == lvl, ]
    n_total <- nrow(sub_data)
    
    if (length(unique(sub_data$outcome)) < 2) {
      cat(sprintf(" - Category [%s]: N = %d, [Warning] Singular outcome, AUC not calculable\n", 
                  labels_dict[as.character(lvl)], n_total))
      next
    }
    
    roc_obj <- pROC::roc(sub_data$outcome, sub_data$prob, quiet = TRUE)
    ci_obj <- pROC::ci.auc(roc_obj, method = "delong") 
    
    cat(sprintf(" - Category [%s]: N = %d | AUC = %.3f (95%% CI: %.3f - %.3f)\n",
                labels_dict[as.character(lvl)], n_total, roc_obj$auc, ci_obj[1], ci_obj[3]))
    
    roc_list[[as.character(lvl)]] <- roc_obj
  }
  
  # Execute DeLong test if exactly two subgroups exist
  if (length(roc_list) == 2) {
    delong_test <- pROC::roc.test(roc_list[[1]], roc_list[[2]], method = "delong")
    cat(sprintf("   => DeLong Test P-value (AUC difference): %.3f\n", delong_test$p.value))
  }
}

# Function to generate formal subgroup AUC table with DeLong P-values
get_subgroup_auc_table <- function(data, subgroup_col, group_name, labels_dict) {
  levels_val <- sort(unique(na.omit(data[[subgroup_col]])))
  res_list <- list()
  roc_list <- list() 
  
  for (lvl in levels_val) {
    sub_data <- data[data[[subgroup_col]] == lvl, ]
    n_total <- nrow(sub_data)
    
    if (length(unique(sub_data$outcome)) < 2) {
      auc_str <- "Not calculable"
    } else {
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
  
  if (length(roc_list) == 2) {
    delong_test <- pROC::roc.test(roc_list[[1]], roc_list[[2]], method = "delong")
    p_val <- delong_test$p.value
    p_val_str <- ifelse(p_val < 0.001, "<0.001", sprintf("%.3f", p_val))
    df_res$`DeLong P-value` <- c(p_val_str, "") 
  } else {
    df_res$`DeLong P-value` <- rep("-", nrow(df_res))
  }
  
  return(df_res)
}

# Define dictionaries and execute console prints
dict_overlap <- c("0" = "Classic AIH", "1" = "AIH-PBC Overlap")
dict_cirrhosis <- c("0" = "Non-Cirrhotic", "1" = "Cirrhotic")
dict_failure <- c("0" = "Without Liver Failure", "1" = "With Liver Failure")

cat("\n============================================================")
cat("\n==== Test 2: Subgroup Performance Analysis (w/ DeLong) ====")
cat("\n============================================================\n")

calc_subgroup_auc(sub_df_test2, "Overlap", "AIH vs. AIH-PBC Overlap", dict_overlap)
calc_subgroup_auc(sub_df_test2, "Cirrhosis", "Cirrhosis Status", dict_cirrhosis)
calc_subgroup_auc(sub_df_test2, "LiverFailure", "Liver Failure Status", dict_failure)

# Consolidate results and export to CSV
tab_overlap_t2 <- get_subgroup_auc_table(sub_df_test2, "Overlap", "Target Population", dict_overlap)
tab_cirrhosis_t2 <- get_subgroup_auc_table(sub_df_test2, "Cirrhosis", "Cirrhosis Status", dict_cirrhosis)
tab_failure_t2 <- get_subgroup_auc_table(sub_df_test2, "LiverFailure", "Liver Failure Status", dict_failure)

supp_table_test2 <- rbind(tab_overlap_t2, tab_cirrhosis_t2, tab_failure_t2)

# Optimize formatting: Empty variable names on alternating rows for clean merging in Word
if(nrow(supp_table_test2) == 6) {
  supp_table_test2$Variable[c(2, 4, 6)] <- ""
}

file_name_t2 <- "Supplementary_Table_Subgroup_AUC_Test2_DeLong.csv"
write.csv(supp_table_test2, file_name_t2, row.names = FALSE, fileEncoding = "UTF-8")

cat("\n[Success] Test 2 subgroup analysis table exported.\n")
cat(sprintf("File: %s\n", file_name_t2))