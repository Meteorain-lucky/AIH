library(ComplexHeatmap)
library(RColorBrewer)
library(circlize)
library(ROSE)
library(scales)
library(lightgbm) 
library(plotROC)
library(pROC)
library(ggplot2)
library(kernelshap)  
library(caret)
library(shapviz) 
library(gbm)
library(rmda)
library(ranger)
library(dplyr) 
library(glmnet)

setwd("/Users/zhoushujie/Desktop")
data = read.csv("data_complete new.csv",header = T, encoding = "GBK")

data$outcome = factor(data$outcome,levels = c(0,1),labels = c('improvement','deteriorate'))

set.seed(52)
inTrain = createDataPartition(y=data[,"outcome"], p=0.7, list=F)
traindata = data[inTrain,]
testdata = data[-inTrain,]

stopifnot("outcome" %in% colnames(traindata))
y_train <- traindata$outcome
y_test  <- testdata$outcome
# ===============================
# Feature Encoding
# ===============================

Xtr_raw <- traindata %>% select(-outcome)
Xte_raw <- testdata  %>% select(-outcome)

Xtr_raw <- Xtr_raw %>% mutate(ANA = as.factor(ANA))
Xte_raw <- Xte_raw %>% mutate(ANA = as.factor(ANA))

# One-hot encoding 
dv <- caret::dummyVars(~ ., data = Xtr_raw, fullRank = TRUE)

X_train <- as.data.frame(predict(dv, newdata = Xtr_raw))
X_test  <- as.data.frame(predict(dv, newdata = Xte_raw))

train_y <- factor(ifelse(y_train == "deteriorate", "fail", "improve"), levels=c("fail","improve"))
test_y  <- factor(ifelse(y_test  == "deteriorate", "fail", "improve"), levels=c("fail","improve"))

ctrl_cv <- trainControl(
  method="repeatedcv", number=5, repeats=5,
  classProbs=TRUE, summaryFunction=twoClassSummary,
  savePredictions="final"
)

# ============================================================
# 3) Build models with SAME fixed features
# ============================================================
feature_names <- c(
  "Child_push_score","TBA","ALP","IgM","ALT","TBil","GGT"
)

train_m <- data.frame(
  outcome = train_y,
  X_train[, feature_names, drop = FALSE]
)

test_m <- data.frame(
  outcome = test_y,
  X_test[, feature_names, drop = FALSE]
)

models <- list()

# --------- LR (Baseline Logistic) ----------
models$LR <- train(outcome ~ ., data = train_m, method = "glm", family = binomial(),
                   metric = "ROC", trControl = ctrl_cv)

# --------- LASSO Logistic (glmnet) ----------
models$LASSO_Logit <- train(
  outcome ~ .,
  data = train_m,
  method = "glmnet",
  family = "binomial",
  metric = "ROC",
  trControl = ctrl_cv,
  preProcess = c("center", "scale"),
  tuneGrid = expand.grid(
    alpha = 1,
    lambda = exp(seq(-6, 1, length.out = 60))
  )
)

# --------- Elastic Net Logistic (glmnet) ----------
models$ENet_Logit <- train(
  outcome ~ .,
  data = train_m,
  method = "glmnet",
  family = "binomial",
  metric = "ROC",
  trControl = ctrl_cv,
  preProcess = c("center", "scale"),
  tuneGrid = expand.grid(
    alpha = c(0.25, 0.5, 0.75),
    lambda = exp(seq(-6, 1, length.out = 60))
  )
)

# --------- RF (Random Forest via ranger) ----------
library(ranger)
library(pROC)

# Strict CV control with down-sampling for class imbalance
ctrl_cv_rf <- trainControl(
  method = "repeatedcv", number = 5, repeats = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final",
  sampling = "down"   
)

# Tuning grid restricting tree complexity
p <- ncol(train_m) - 1
rf_grid <- expand.grid(
  mtry = unique(pmax(1, round(c(sqrt(p), p/3, p/2)))),
  splitrule = c("gini"),
  min.node.size = c(5, 10, 20)   
)

set.seed(52)
models$RF <- train(
  outcome ~ .,
  data = train_m,
  method = "ranger",
  metric = "ROC",
  trControl = ctrl_cv_rf,
  tuneGrid = rf_grid,
  num.trees = 1200,
  importance = "impurity",
  sample.fraction = 0.70
)

# --------- DT (Decision Tree) ----------
models$DT <- train(outcome ~ ., data = train_m, method = "rpart",
                   metric = "ROC", trControl = ctrl_cv, tuneLength = 10)

# --------- SVM (Radial Basis Function) ----------
models$SVM <- train(outcome ~ ., data = train_m, method = "svmRadial",
                    metric = "ROC", trControl = ctrl_cv,
                    preProcess = c("center", "scale"), tuneLength = 10)

# --------- MLP (Neural Network) ----------
models$MLP <- train(outcome ~ ., data = train_m, method = "nnet",
                    metric = "ROC", trControl = ctrl_cv,
                    preProcess = c("center", "scale"), tuneLength = 10, trace = FALSE)

# --------- GBM (Gradient Boosting Machine) ----------
models$GBM <- train(outcome ~ ., data = train_m, method = "gbm",
                    metric = "ROC", trControl = ctrl_cv, tuneLength = 10, verbose = FALSE)

# --------- KNN (K-Nearest Neighbors) ----------
models$KNN <- train(outcome ~ ., data = train_m, method = "knn",
                    metric = "ROC", trControl = ctrl_cv,
                    preProcess = c("center", "scale"), tuneLength = 15)

# ---------- Naive Bayes ----------
cat("Training Naive Bayes model...\n")
tryCatch({
  models$NB <- train(
    outcome ~ .,
    data = train_m,
    method = "naive_bayes",
    metric = "ROC",
    trControl = ctrl_cv,
    preProcess = c("center", "scale"),
    tuneLength = 3
  )
  cat("   NB completed, CV ROC:", round(max(models$NB$results$ROC, na.rm = TRUE), 3), "\n")
}, error = function(e) {
  cat("   NB failed:", e$message, "\n")
})


# ---------- LightGBM ----------
cat("\nTraining LightGBM (binary, fail=1)...\n")
suppressPackageStartupMessages({
  library(lightgbm)
  library(pROC)
  library(dplyr)
})

set.seed(52)
stopifnot(is.factor(train_m$outcome))

# Ensure correct outcome levels
train_m$outcome <- factor(train_m$outcome, levels = c("fail", "improve"))
test_m$outcome  <- factor(test_m$outcome,  levels = c("fail", "improve"))
y_all <- ifelse(train_m$outcome == "fail", 1, 0)

X_all <- train_m %>% dplyr::select(-outcome) %>% dplyr::mutate(dplyr::across(dplyr::everything(), ~ as.numeric(.))) %>% as.matrix()
X_test <- test_m %>% dplyr::select(-outcome) %>% dplyr::mutate(dplyr::across(dplyr::everything(), ~ as.numeric(.))) %>% as.matrix()

# Split a validation set from training data for early stopping
idx <- sample(seq_len(nrow(X_all)))
n_valid <- max(1, round(0.2 * nrow(X_all)))
valid_idx <- idx[1:n_valid]
train_idx <- idx[(n_valid+1):length(idx)]

dtrain <- lgb.Dataset(data = X_all[train_idx, , drop = FALSE], label = y_all[train_idx])
dvalid <- lgb.Dataset(data = X_all[valid_idx, , drop = FALSE], label = y_all[valid_idx])

# Define parameters to control overfitting
params <- list(
  objective = "binary", metric = "auc", boosting = "gbdt",
  learning_rate = 0.03, num_leaves = 15, max_depth = 4,
  min_data_in_leaf = 10, feature_fraction = 0.9, bagging_fraction = 0.8,
  bagging_freq = 1, lambda_l2 = 1.0, verbose = -1
)

lgb_model <- lgb.train(
  params = params, data = dtrain, nrounds = 2000,
  valids = list(valid = dvalid), early_stopping_rounds = 50, verbose = 50
)

pred_train_all <- predict(lgb_model, X_all)
pred_test <- predict(lgb_model, X_test)

models$LGBM <- list(
  method = "lightgbm",
  model = lgb_model,
  pred_train = pred_train_all,
  pred_test  = pred_test,
  features = colnames(X_all)
)

# --------- XGBoost (native) ----------
cat("\nTraining XGBoost (native)...\n")
suppressPackageStartupMessages({
  library(xgboost)
  library(caret)
})

train_m$outcome <- factor(train_m$outcome, levels = c("fail", "improve"))
test_m$outcome  <- factor(test_m$outcome,  levels = c("fail", "improve"))
Xtr <- as.matrix(train_m[, -1, drop = FALSE])
Xte <- as.matrix(test_m[,  -1, drop = FALSE])
ytr <- ifelse(train_m$outcome == "fail", 1L, 0L)
yte <- ifelse(test_m$outcome  == "fail", 1L, 0L)

dtrain <- xgb.DMatrix(data = Xtr, label = ytr)
dtest  <- xgb.DMatrix(data = Xte, label = yte)

# Stratified folds
set.seed(52)
folds <- caret::createFolds(train_m$outcome, k = 5, returnTrain = FALSE)

xgb_params <- list(
  booster = "gbtree", objective = "binary:logistic", eval_metric = "auc",
  max_depth = 2, eta = 0.03, min_child_weight = 3,
  subsample = 0.80, colsample_bytree = 0.70, gamma = 0,
  reg_lambda = 1.0, reg_alpha  = 0.1 
)

set.seed(52)
cv <- xgb.cv(
  params = xgb_params, data = dtrain, nrounds = 2000,
  folds = folds, early_stopping_rounds = 30, verbose = 0
)

best_row <- which.max(cv$evaluation_log$test_auc_mean)
best_nrounds <- cv$evaluation_log$iter[best_row]

final_xgb <- xgb.train(params = xgb_params, data = dtrain, nrounds = best_nrounds, verbose = 0)

models$XGB_native <- list(
  method = "xgboost_native", model = final_xgb, params = xgb_params,
  best_nrounds = best_nrounds, features = colnames(Xtr)
)

# ============================================================
# CR-SCAD (SCAD-penalized Logistic Regression)
# ============================================================
cat("\nTraining CR-SCAD (SCAD-penalized Logistic Regression)...\n")
suppressPackageStartupMessages({
  if (!requireNamespace("ncvreg", quietly = TRUE)) {
    # Use HTTP mirror to bypass potential SSL issues
    options(repos = c(CRAN = "http://mirrors.tuna.tsinghua.edu.cn/CRAN/"))
    install.packages("ncvreg")
  }
  library(ncvreg)
})

tryCatch({
  Xtr_scad <- as.matrix(train_m[, feature_names, drop = FALSE])
  ytr_scad <- ifelse(train_m$outcome == "fail", 1, 0)
  
  set.seed(52)
  cv_scad <- cv.ncvreg(X = Xtr_scad, y = ytr_scad, family = "binomial", penalty = "SCAD", nfolds = 5)
  
  models$CR_SCAD <- list(
    method = "cr_scad",
    model = cv_scad,
    best_lambda = cv_scad$lambda.min,
    features = feature_names
  )
  cat("   CR-SCAD completed and added to models list.\n")
}, error = function(e) {
  cat("\n[Warning] CR-SCAD training failed:", e$message, "\n")
})
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