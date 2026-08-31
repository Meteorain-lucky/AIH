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
data = read.csv("data_complete.csv",header = T, encoding = "GBK")

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

######3) Build models with SAME fixed features
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


library(nricens)
library(pROC)

# ============================================================
# Incremental Clinical Value Analysis (Continuous NRI & Bootstrap IDI)
# ============================================================
cat("\n\n============================================================")
cat("\n==== Incremental Value Analysis ====")
cat("\n============================================================\n")

# --- Data preparation ---
prob_new_test <- get_prob_fail(models$GBM, "GBM", test_m, outcome_col = "outcome")
test_y_numeric <- ifelse(test_m$outcome == "fail", 1, 0)
n_samples <- length(test_y_numeric)
B_boot <- 1000 
set.seed(123)  

events <- test_y_numeric == 1    
nonevents <- test_y_numeric == 0 

# ------------------------------------------------------------
# Module A: Head-to-head vs Child-Pugh Score
# ------------------------------------------------------------
cp_col_name <- "Child_push_score" 

if (cp_col_name %in% colnames(traindata)) {
  cat("\n>>> [ 1. Vs Child-Pugh Score (Head-to-head) ] <<<\n")
  
  train_cp_df <- data.frame(outcome = train_y, CP_Score = traindata[[cp_col_name]])
  test_cp_df  <- data.frame(outcome = test_y,  CP_Score = testdata[[cp_col_name]])
  base_model_cp <- glm(outcome ~ CP_Score, data = train_cp_df, family = binomial)
  prob_base_test_cp <- predict(base_model_cp, newdata = test_cp_df, type = "response")
  
  p_new_cp <- prob_new_test 
  
  # --- Continuous NRI ---
  nri_cont_cp <- nribin(event = test_y_numeric, p.std = prob_base_test_cp, p.new = p_new_cp, 
                        cut = 0, updown = 'diff', msg = FALSE)
  p_val_cont_cp <- 2 * (1 - pnorm(abs(nri_cont_cp$nri["NRI", "Estimate"] / nri_cont_cp$nri["NRI", "Std.Error"])))
  
  cat("\n--- (1) Continuous NRI ---\n")
  print(nri_cont_cp$nri)
  cat(sprintf("-> Continuous NRI p-value = %e\n", p_val_cont_cp))
  
  # --- Bootstrap IDI ---
  IDI_value_cp <- (mean(p_new_cp[events]) - mean(prob_base_test_cp[events])) - 
    (mean(p_new_cp[nonevents]) - mean(prob_base_test_cp[nonevents]))
  
  idi_boot_cp <- numeric(B_boot)
  for(i in 1:B_boot) {
    boot_idx <- sample(1:n_samples, n_samples, replace = TRUE)
    b_y <- test_y_numeric[boot_idx]
    b_ev <- b_y == 1; b_nev <- b_y == 0
    if(sum(b_ev) > 0 && sum(b_nev) > 0) {
      idi_boot_cp[i] <- (mean(p_new_cp[boot_idx][b_ev]) - mean(prob_base_test_cp[boot_idx][b_ev])) - 
        (mean(p_new_cp[boot_idx][b_nev]) - mean(prob_base_test_cp[boot_idx][b_nev]))
    } else { idi_boot_cp[i] <- NA }
  }
  idi_se_cp <- sd(idi_boot_cp, na.rm = TRUE)
  idi_pval_cp <- 2 * (1 - pnorm(abs(IDI_value_cp / idi_se_cp)))
  
  cat("\n--- (2) Integrated Discrimination Improvement (IDI) ---\n")
  cat(sprintf("IDI: %.4f (%.2f%%)\n", IDI_value_cp, IDI_value_cp * 100))
  cat(sprintf("Bootstrap 95%% CI: [%.4f, %.4f]\n", IDI_value_cp - 1.96*idi_se_cp, IDI_value_cp + 1.96*idi_se_cp))
  cat(sprintf("-> IDI p-value = %e\n", idi_pval_cp))
} else {
  cat(sprintf("\n[Warning] Column '%s' not found, skipping analysis.\n", cp_col_name))
}
cat("\n------------------------------------------------------------\n")

# ------------------------------------------------------------
# Shared Baseline for MELD Models
# ------------------------------------------------------------
meld_col_name <- "Admission_MELD_score" 
if (meld_col_name %in% colnames(traindata)) {
  
  train_meld_df <- data.frame(outcome = train_y_numeric, MELD_Score = traindata[[meld_col_name]], GBM_Prob = prob_gbm_train)
  test_meld_df  <- data.frame(outcome = test_y_numeric,  MELD_Score = testdata[[meld_col_name]],  GBM_Prob = prob_gbm_test)
  
  base_model_meld <- glm(outcome ~ MELD_Score, data = train_meld_df, family = binomial)
  prob_base_test_meld <- predict(base_model_meld, newdata = test_meld_df, type = "response")
  
  # ------------------------------------------------------------
  # Module B: Head-to-head vs MELD Score
  # ------------------------------------------------------------
  cat(">>> [ 2. Vs MELD Score (Head-to-head) ] <<<\n")
  
  p_new_meld_h2h <- prob_new_test 
  
  # --- Continuous NRI ---
  nri_cont_meld_h2h <- nribin(event = test_y_numeric, p.std = prob_base_test_meld, p.new = p_new_meld_h2h, 
                              cut = 0, updown = 'diff', msg = FALSE)
  p_val_cont_m_h2h <- 2 * (1 - pnorm(abs(nri_cont_meld_h2h$nri["NRI", "Estimate"] / nri_cont_meld_h2h$nri["NRI", "Std.Error"])))
  
  cat("\n--- (1) Continuous NRI ---\n")
  print(nri_cont_meld_h2h$nri)
  cat(sprintf("-> Continuous NRI p-value = %e\n", p_val_cont_m_h2h))
  
  # --- Bootstrap IDI ---
  IDI_value_meld_h2h <- (mean(p_new_meld_h2h[events]) - mean(prob_base_test_meld[events])) - 
    (mean(p_new_meld_h2h[nonevents]) - mean(prob_base_test_meld[nonevents]))
  
  idi_boot_meld_h2h <- numeric(B_boot)
  for(i in 1:B_boot) {
    boot_idx <- sample(1:n_samples, n_samples, replace = TRUE)
    b_y <- test_y_numeric[boot_idx]
    b_ev <- b_y == 1; b_nev <- b_y == 0
    if(sum(b_ev) > 0 && sum(b_nev) > 0) {
      idi_boot_meld_h2h[i] <- (mean(p_new_meld_h2h[boot_idx][b_ev]) - mean(prob_base_test_meld[boot_idx][b_ev])) - 
        (mean(p_new_meld_h2h[boot_idx][b_nev]) - mean(prob_base_test_meld[boot_idx][b_nev]))
    } else { idi_boot_meld_h2h[i] <- NA }
  }
  idi_se_meld_h2h <- sd(idi_boot_meld_h2h, na.rm = TRUE)
  idi_pval_meld_h2h <- 2 * (1 - pnorm(abs(IDI_value_meld_h2h / idi_se_meld_h2h)))
  
  cat("\n--- (2) Integrated Discrimination Improvement (IDI) ---\n")
  cat(sprintf("IDI: %.4f (%.2f%%)\n", IDI_value_meld_h2h, IDI_value_meld_h2h * 100))
  cat(sprintf("Bootstrap 95%% CI: [%.4f, %.4f]\n", IDI_value_meld_h2h - 1.96*idi_se_meld_h2h, IDI_value_meld_h2h + 1.96*idi_se_meld_h2h))
  cat(sprintf("-> IDI p-value = %e\n", idi_pval_meld_h2h))
  
  cat("\n------------------------------------------------------------\n")
  
  # ------------------------------------------------------------
  # Module C: Incremental Value vs MELD (Multivariable Adjusted Framework)
  # ------------------------------------------------------------
  cat(">>> [ 3. Incremental Value vs MELD (Multivariable Adjusted) ] <<<\n")
  
  adj_model_meld <- glm(outcome ~ MELD_Score + GBM_Prob, data = train_meld_df, family = binomial)
  prob_adj_test_meld <- predict(adj_model_meld, newdata = test_meld_df, type = "response")
  
  gbm_pval_meld <- summary(adj_model_meld)$coefficients["GBM_Prob", "Pr(>|z|)"]
  cat(sprintf("-> Multivariable Analysis: GBM Probability independent P-value = %e\n", gbm_pval_meld))
  
  p_new_meld_adj <- prob_adj_test_meld
  
  # --- Continuous NRI ---
  nri_cont_meld_adj <- nribin(event = test_y_numeric, p.std = prob_base_test_meld, p.new = p_new_meld_adj, 
                              cut = 0, updown = 'diff', msg = FALSE)
  p_val_cont_m_adj <- 2 * (1 - pnorm(abs(nri_cont_meld_adj$nri["NRI", "Estimate"] / nri_cont_meld_adj$nri["NRI", "Std.Error"])))
  
  cat("\n--- (1) Continuous NRI ---\n")
  print(nri_cont_meld_adj$nri)
  cat(sprintf("-> Continuous NRI p-value = %e\n", p_val_cont_m_adj))
  
  # --- Bootstrap IDI ---
  IDI_value_meld_adj <- (mean(p_new_meld_adj[events]) - mean(prob_base_test_meld[events])) - 
    (mean(p_new_meld_adj[nonevents]) - mean(prob_base_test_meld[nonevents]))
  
  idi_boot_meld_adj <- numeric(B_boot)
  for(i in 1:B_boot) {
    boot_idx <- sample(1:n_samples, n_samples, replace = TRUE)
    b_y <- test_y_numeric[boot_idx]
    b_ev <- b_y == 1; b_nev <- b_y == 0
    if(sum(b_ev) > 0 && sum(b_nev) > 0) {
      idi_boot_meld_adj[i] <- (mean(p_new_meld_adj[boot_idx][b_ev]) - mean(prob_base_test_meld[boot_idx][b_ev])) - 
        (mean(p_new_meld_adj[boot_idx][b_nev]) - mean(prob_base_test_meld[boot_idx][b_nev]))
    } else { idi_boot_meld_adj[i] <- NA }
  }
  idi_se_meld_adj <- sd(idi_boot_meld_adj, na.rm = TRUE)
  idi_pval_meld_adj <- 2 * (1 - pnorm(abs(IDI_value_meld_adj / idi_se_meld_adj)))
  
  cat("\n--- (2) Integrated Discrimination Improvement (IDI) ---\n")
  cat(sprintf("IDI: %.4f (%.2f%%)\n", IDI_value_meld_adj, IDI_value_meld_adj * 100))
  cat(sprintf("Bootstrap 95%% CI: [%.4f, %.4f]\n", IDI_value_meld_adj - 1.96*idi_se_meld_adj, IDI_value_meld_adj + 1.96*idi_se_meld_adj))
  cat(sprintf("-> IDI p-value = %e\n", idi_pval_meld_adj))
  
} else {
  cat(sprintf("\n[Warning] Column '%s' not found, skipping analysis.\n", meld_col_name))
}

# ============================================================
# Module D: Change in AUC & DeLong's Test (All Comparisons)
# ============================================================
cat("\n\n============================================================")
cat("\n==== Change in AUC & DeLong's Test ====")
cat("\n============================================================\n")

roc_gbm_pure <- pROC::roc(test_y_numeric, prob_new_test, quiet = TRUE)

if (exists("p_new_cp") && exists("prob_base_test_cp")) {
  roc_cp_base <- pROC::roc(test_y_numeric, prob_base_test_cp, quiet = TRUE)
  
  auc_diff_cp <- as.numeric(roc_gbm_pure$auc) - as.numeric(roc_cp_base$auc)
  delong_test_cp <- pROC::roc.test(roc_cp_base, roc_gbm_pure, method = "delong")
  
  cat(">>> [ 1. AUC Improvement vs Child-Pugh (Head-to-Head) ] <<<\n")
  cat(sprintf("Baseline (Child-Pugh) AUC: %.3f\n", roc_cp_base$auc))
  cat(sprintf("New (Pure GBM) AUC:        %.3f\n", roc_gbm_pure$auc))
  cat(sprintf("Change in AUC (ΔAUC):      %+.3f\n", auc_diff_cp))
  cat(sprintf("DeLong Test p-value:       %e\n\n", delong_test_cp$p.value))
}

if (exists("prob_base_test_meld")) {
  roc_meld_base <- pROC::roc(test_y_numeric, prob_base_test_meld, quiet = TRUE)
  
  # Head-to-Head DeLong
  auc_diff_meld_h2h <- as.numeric(roc_gbm_pure$auc) - as.numeric(roc_meld_base$auc)
  delong_test_meld_h2h <- pROC::roc.test(roc_meld_base, roc_gbm_pure, method = "delong")
  
  cat(">>> [ 2. AUC Improvement vs MELD (Head-to-Head) ] <<<\n")
  cat(sprintf("Baseline (MELD Only) AUC:  %.3f\n", roc_meld_base$auc))
  cat(sprintf("New (Pure GBM) AUC:        %.3f\n", roc_gbm_pure$auc))
  cat(sprintf("Change in AUC (ΔAUC):      %+.3f\n", auc_diff_meld_h2h))
  cat(sprintf("DeLong Test p-value:       %e\n\n", delong_test_meld_h2h$p.value))
  
  # Adjusted Framework DeLong
  if (exists("p_new_meld_adj")) {
    roc_meld_adj <- pROC::roc(test_y_numeric, p_new_meld_adj, quiet = TRUE)
    
    auc_diff_meld_adj <- as.numeric(roc_meld_adj$auc) - as.numeric(roc_meld_base$auc)
    delong_test_meld_adj <- pROC::roc.test(roc_meld_base, roc_meld_adj, method = "delong")
    
    cat(">>> [ 3. AUC Improvement vs MELD (Adjusted Framework) ] <<<\n")
    cat(sprintf("Baseline (MELD Only) AUC:  %.3f\n", roc_meld_base$auc))
    cat(sprintf("Adjusted (MELD + GBM) AUC: %.3f\n", roc_meld_adj$auc))
    cat(sprintf("Change in AUC (ΔAUC):      %+.3f\n", auc_diff_meld_adj))
    cat(sprintf("DeLong Test p-value:       %e\n", delong_test_meld_adj$p.value))
    
    cat("\n=== Multivariable Adjusted Model (MELD + GBM) Weights ===\n")
    print(summary(adj_model_meld)$coefficients)
  }
}
cat("============================================================\n")

