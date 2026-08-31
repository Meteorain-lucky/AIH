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
data = read.csv("data_complete（type）.csv",header = T, encoding = "GBK")

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
#############Build models with SAME fixed features
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

# ==========================================
# Baseline Predictors Correlation Analysis
# ==========================================

# Specify baseline variables to check
vars_to_check <- c("Child_push_score","TBA","ALP","IgM","ALT","TBil","GGT") 

# Convert outcome to numeric (0/1) for correlation calculation
numeric_outcome <- ifelse(data$outcome == "deteriorate", 1, 0)

# Calculate Pearson correlation between baseline predictors and binary outcome
cor_results <- sapply(data[vars_to_check], function(x) {
  cor(x, numeric_outcome, use = "complete.obs", method = "pearson")
})

print("Correlation coefficients between baseline predictors and outcome:")
print(cor_results)

# Statistical significance test for specific variables (e.g., ALT)
cor_alt_test <- cor.test(data$ALT, numeric_outcome, method = "pearson")
print(cor_alt_test)

p_values <- sapply(data[vars_to_check], function(x) {
  cor.test(x, numeric_outcome, use = "complete.obs", method = "pearson")$p.value
})


final_cor_table <- data.frame(
  Correlation_r = cor_results,
  P_value = p_values
)

print("=== 7个变量的完整相关性分析结果 ===")
print(final_cor_table)

# ==========================================
# Outcome-Component Analysis
# ==========================================
library(dplyr)
library(ggplot2)

# ==========================================
# 1. Data preparation and grouping
# ==========================================
df_failed <- data_new %>% 
  filter(type %in% c(1, 2, 3)) %>%
  mutate(Event_Category = case_when(
    type %in% c(1, 3) ~ "Hard Events (N=31)",
    type == 2 ~ "Biochemical Only (N=146)",
    TRUE ~ "Other"
  ))

# Adjust factor levels for plotting
df_failed$Event_Category <- factor(df_failed$Event_Category, 
                                   levels = c("Biochemical Only (N=146)", 
                                              "Hard Events (N=31)"))

# ==========================================
# 2. Summary statistics of predicted probabilities
# ==========================================
summary_stats <- df_failed %>%
  group_by(Event_Category) %>%
  summarise(
    Count = n(),
    Median_Prob = median(Predicted_Prob, na.rm = TRUE),
    Q1 = quantile(Predicted_Prob, 0.25, na.rm = TRUE),
    Q3 = quantile(Predicted_Prob, 0.75, na.rm = TRUE)
  )

cat("\n============== Summary Statistics ==============\n")
print(summary_stats)
cat("==============================================\n\n")

# ==========================================
# 3. Statistical testing and effect size
# ==========================================
# Extract predicted probabilities for both groups
prob_bio <- df_failed$Predicted_Prob[df_failed$Event_Category == "Biochemical Only (N=146)"]
prob_hard <- df_failed$Predicted_Prob[df_failed$Event_Category == "Hard Events (N=31)"]

# A. Wilcoxon rank-sum test
wilcox_res <- wilcox.test(Predicted_Prob ~ Event_Category, data = df_failed)
p_val <- wilcox_res$p.value

# B. Cohen's d effect size
n1 <- length(prob_bio)
n2 <- length(prob_hard)
v1 <- var(prob_bio, na.rm=TRUE)
v2 <- var(prob_hard, na.rm=TRUE)
pooled_sd <- sqrt(((n1-1)*v1 + (n2-1)*v2) / (n1+n2-2))
cohens_d <- abs(mean(prob_hard, na.rm=TRUE) - mean(prob_bio, na.rm=TRUE)) / pooled_sd

# C. Kolmogorov-Smirnov (KS) test
ks_res <- ks.test(prob_bio, prob_hard)
p_val_ks <- ks_res$p.value

cat("============== Statistical Metrics ==============\n")
cat(sprintf("Wilcoxon P-value    : %.4f\n", p_val))
cat(sprintf("Cohen's d           : %.4f\n", cohens_d))
cat(sprintf("KS Test P-value     : %.4f\n", p_val_ks))
cat("===============================================\n")

# ==========================================
# 4. Visualization: Violin plot + Boxplot
# ==========================================
# Format P-value for plotting
p_label <- ifelse(p_val < 0.001, "P < 0.001", sprintf("P = %.3f", p_val))

ggplot(df_failed, aes(x = Event_Category, y = Predicted_Prob, fill = Event_Category)) +
  # Violin plot
  geom_violin(alpha = 0.3, trim = FALSE, color = NA) +
  # Boxplot
  geom_boxplot(width = 0.2, alpha = 0.9, outlier.shape = NA, color = "black") +
  # Jitter points
  geom_jitter(width = 0.1, size = 1.5, alpha = 0.6, color = "darkgray") +
  scale_fill_manual(values = c("Hard Events (N=31)" = "#E69F00", 
                               "Biochemical Only (N=146)" = "#56B4E9")) +
  theme_minimal() +
  labs(
    title = "Probability Distribution by Failure Subtype",
    subtitle = paste0("Wilcoxon test: ", p_label, " | Cohen's d: ", round(cohens_d, 3)),
    x = "Subtype of Treatment Failure",
    y = "Predicted Probability of Failure"
  ) +
  theme(
    legend.position = "none",
    plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
    plot.subtitle = element_text(hjust = 0.5, face = "italic", color = "darkred", size = 11),
    axis.text = element_text(size = 11),
    axis.title = element_text(size = 12)
  )
