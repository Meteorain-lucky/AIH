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

getwd()
setwd("/Users/zhoushujie/Desktop")

if (.Platform$OS.type == "windows") {
  windowsFonts(Times = windowsFont("Times New Roman"))  
}
par(family = "Times")
data = read.csv("original data.csv",header = T, encoding = "GBK")

# ============================================================
# Step 1: Data Preprocessing
# ============================================================

# Batch convert categorical variables to factors
data[,1:24] <- lapply(data[,1:24], as.factor) 
str(data) 
summary(data) 

continuous_vars <- 25:ncol(data)

# Visualizing Missing Values
data %>%
  gg_miss_var(show_pct = TRUE) +  
  theme_bw(base_size = 16, base_family = "Times") 

# Saving Missing Value Ratio
missing_ratio <- colMeans(is.na(data))
missing_ratio_df <- data.frame(Variable = names(missing_ratio), Missing_Ratio = missing_ratio)
write.csv(missing_ratio_df, "Proportion of missing values.csv", row.names = FALSE)

# Export Missing Value Plot to PDF
p_missing <- data %>%
  gg_miss_var(show_pct = TRUE) + 
  theme_bw(base_size = 16, base_family = "serif") + 
  labs(title = "Missing Values Distribution")

ggsave("Missing_Values_Plot.pdf", plot = p_missing, width = 8, height = 6)

# Multiple Imputation 
data_imputed <- mice(data, seed = 123, print = FALSE, m = 5)
data_imputed$method 
data_imputed2 <- mice::complete(data_imputed, action = 2)
write.csv(data_imputed2, "data_complete.csv", row.names = FALSE)

# Load imputed data
data = read.csv("data_complete new.csv",header = T, encoding = "GBK")
colMeans(is.na(data)) 

# ============================================================
# Step 2: Train/Test Split
# ============================================================

data$outcome <- factor(data$outcome, levels = c(0, 1), labels = c("Treatment Success", "Treatment Failure"))
set.seed(123)
inTrain = createDataPartition(y=data[,"outcome"], p=0.7, list=F) 
traindata = data[inTrain,] 
testdata = data[-inTrain,] 
write.csv(traindata, "traindata.csv", row.names = F) 
write.csv(testdata, "testdata.csv", row.names = F) 

# ============================================================
# Step 3: Baseline Characteristics Table
# ============================================================

myVars <- colnames(data[, 2:ncol(data)])  
catVars <- colnames(data[, 2:25])         
contVars <- colnames(data[, 26:ncol(data)])  

# Normality Check
noncenter <- c()
for (var in contVars) {
  if (is.numeric(data[[var]])) {
    test_result <- shapiro.test(data[[var]])
    if (test_result$p.value < 0.05) {
      noncenter <- c(noncenter, var)
    }
  } else {
    print(paste(var, "Normal Distribution"))
  }
}

table1 <- CreateTableOne(vars = myVars,  
                         factorVars = catVars,  
                         data = data,  
                         strata='outcome', 
                         addOverall = TRUE)  
table1 <- print(table1,showAllLevels = T,catDigits = 2,contDigits = 2,pDigits =3,nonnormal =noncenter)
write.csv(table1,file = "Table 1.csv") 

# ============================================================
# Modeling and Feature Selection
# ============================================================

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
# Feature Selection
# ============================================================

# ---------- A) Elastic Net (glmnet) ----------
x_mat <- as.matrix(X_train)
best_enet_cv <- cv.glmnet(x_mat, y_train, family="binomial", alpha=0.75, nfolds=5)
coef_enet <- coef(best_enet_cv, s="lambda.min") 

enet_sel <- rownames(coef_enet)[which(as.numeric(coef_enet) != 0)]
enet_sel <- setdiff(enet_sel, "(Intercept)")

cat("\n[ENet selector] Fixed alpha = 0.75 (using lambda.min), selected =", length(enet_sel), "features\n")
print(enet_sel)

# ---------- B) SVM-RFE ----------
nzv <- nearZeroVar(X_train)
if(length(nzv) > 0){
  cat("[SVM-RFE] remove nearZeroVar:", length(nzv), "\n")
  X_train_svm <- X_train[, -nzv, drop=FALSE]
} else {
  X_train_svm <- X_train
}

myTwoClassSummary <- function(data, lev = NULL, model = NULL){
  out_roc <- caret::twoClassSummary(data, lev = lev, model = model)
  out_acc <- caret::defaultSummary(data, lev = lev, model = model)
  c(out_roc, out_acc) 
}

svmRocFuncs <- caretFuncs
svmRocFuncs$summary <- myTwoClassSummary

ctrl_rfe <- rfeControl(
  functions = svmRocFuncs,
  method = "cv",
  number = 5
)

train_y <- factor(ifelse(y_train == "deteriorate", "fail", "improve"),
                  levels = c("fail","improve"))
test_y  <- factor(ifelse(y_test  == "deteriorate", "fail", "improve"),
                  levels = c("fail","improve"))

svm_profile <- rfe(
  x = X_train_svm,
  y = train_y,                     
  sizes = c(3,5,8,10,12,15,20),
  rfeControl = ctrl_rfe,
  method = "svmLinear",
  metric = "ROC",
  preProcess = c("center","scale"),
  trControl = trainControl(
    method="cv", number=5,
    classProbs=TRUE,
    summaryFunction=myTwoClassSummary  
  )
)

svm_sel <- predictors(svm_profile)
cat("[SVM-RFE selector] selected =", length(svm_sel), "\n")
print(svm_sel)

# ---------- C) Boruta ----------
suppressPackageStartupMessages(library(Boruta))
y_bor01 <- factor(
  ifelse(y_train == "deteriorate", 1, 0),
  levels = c(0,1)
)

bor <- Boruta(
  x = X_train,
  y = y_bor01,
  doTrace = 0,
  maxRuns = 500
)

bor_fix <- TentativeRoughFix(bor)
bor_sel <- getSelectedAttributes(bor_fix, withTentative = FALSE)
cat("[Boruta selector] selected =", length(bor_sel), "\n")
print(bor_sel)

# ---------- D) RF importance (ranger) ----------
rf_dat <- data.frame(outcome=factor(y_train), X_train)
rf_fit <- ranger(outcome ~ ., data=rf_dat,
                 probability=TRUE,
                 num.trees=800,
                 importance="impurity",
                 seed=42)
rf_imp <- sort(rf_fit$variable.importance, decreasing=TRUE)
rf_sel <- names(rf_imp)[1:min(20, length(rf_imp))]
cat("[RF importance selector] selected(top) =", length(rf_sel), "\n")
print(rf_sel)

# ---------- Consensus ----------
set_enet <- unique(enet_sel)
set_svm  <- unique(svm_sel)
set_bor  <- unique(bor_sel)
set_rf   <- unique(rf_sel)

all_feats <- colnames(X_train)

sel_tbl <- tibble(feature = all_feats) %>%
  mutate(
    selected_enet = feature %in% set_enet,
    selected_svm  = feature %in% set_svm,
    selected_bor  = feature %in% set_bor,
    selected_rf   = feature %in% set_rf,
    n_methods = selected_enet + selected_svm + selected_bor + selected_rf
  ) %>%
  arrange(desc(n_methods))

consensus_3of4 <- sel_tbl %>%
  filter(n_methods >= 3) %>%
  pull(feature) %>% sort()

cat("\nConsensus(>=3/4):", length(consensus_3of4), "\n")
print(consensus_3of4)

write.csv(sel_tbl, "AIH_feature_selection_table_ENet_SVM_Boruta_RF.csv", row.names=FALSE)
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


# ============================================================
# 4) Unified Evaluation: Calibration, ROC, Confusion Matrix, DCA
# ============================================================
suppressPackageStartupMessages({
  library(pROC)
  library(ggplot2)
  library(dplyr)
  library(tibble)
  library(caret)
  library(cvms)
  library(rmda)
})

# Consistent color mapping for all models
model_colors <- c(
  "LR"          = "#0072B2",  
  "LASSO_Logit" = "#F26B8B",  
  "ENet_Logit"  = "#bcbd22",  
  "RF"          = "#CC79A7",  
  "DT"          = "#56B4E9",  
  "SVM"         = "#E69F00",  
  "MLP"         = "#332288",  
  "GBM"         = "#F0E442",  
  "KNN"         = "#27F53C",  
  "NB"          = "#2ca02c",  
  "LGBM"        = "#d62728",  
  "XGB_native"  = "#9467bd",  
  "CR_SCAD"     = "#8c564b"   
)

# Fallback for missing colors
get_color_map <- function(model_names){
  cm <- model_colors[intersect(names(model_colors), model_names)]
  missing <- setdiff(model_names, names(cm))
  if(length(missing) > 0){
    extra <- grDevices::rainbow(length(missing))
    names(extra) <- missing
    cm <- c(cm, extra)
  }
  cm
}

Train_df <- train_m
Test_df  <- test_m

# Standardize outcome levels (improve=0, fail=1)
Train_df$outcome <- factor(Train_df$outcome, levels = c("improve", "fail"))
Test_df$outcome  <- factor(Test_df$outcome,  levels = c("improve", "fail"))

# -----------------------------
# 4.1 Helper: Get Predicted Probability of 'fail'
# -----------------------------
get_prob_fail <- function(model_obj, model_name, newdata, outcome_col = "outcome") {
  stopifnot(is.data.frame(newdata))
  X <- newdata %>% dplyr::select(-dplyr::all_of(outcome_col))
  
  # A) caret::train
  if (inherits(model_obj, "train")) {
    p <- predict(model_obj, newdata = X, type = "prob")
    return(as.numeric(p[, "fail"]))
  }
  
  # B) LightGBM wrapper
  if (is.list(model_obj) && !is.null(model_obj$method) && model_obj$method == "lightgbm") {
    Xmat <- as.matrix(X[, model_obj$features, drop = FALSE])
    p <- predict(model_obj$model, Xmat)
    return(as.numeric(p))
  }
  
  # C) XGBoost native wrapper
  if (is.list(model_obj) && !is.null(model_obj$method) && model_obj$method == "xgboost_native") {
    Xmat <- as.matrix(X[, model_obj$features, drop = FALSE])
    d <- xgboost::xgb.DMatrix(data = Xmat)
    p <- predict(model_obj$model, d)
    return(as.numeric(p))
  }
  
  # D) CR-SCAD wrapper
  if (is.list(model_obj) && !is.null(model_obj$method) && model_obj$method == "cr_scad") {
    Xmat <- as.matrix(X[, model_obj$features, drop = FALSE])
    p <- predict(model_obj$model, X = Xmat, type = "response", lambda = model_obj$best_lambda)
    return(as.numeric(p))
  }
  
  stop(paste0("Unsupported model type for: ", model_name))
}

# -----------------------------
# 4.2 Build Prediction Tables
# -----------------------------
pred_train <- data.frame(outcome = Train_df$outcome)
pred_test  <- data.frame(outcome = Test_df$outcome)

for (mn in names(models)) {
  cat("[Predicting probabilities] ", mn, "\n")
  pred_train[[mn]] <- get_prob_fail(models[[mn]], mn, Train_df, outcome_col = "outcome")
  pred_test[[mn]]  <- get_prob_fail(models[[mn]], mn, Test_df,  outcome_col = "outcome")
}

datalist <- list(Train = pred_train, Test = pred_test)

# ============================================================
# 4.3 Calibration Curve
# ============================================================
cutpoint <- 5

for (tt in names(datalist)) {
  newdata <- datalist[[tt]]
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
      axis.title = element_text(size = 12, face = "bold"),
      axis.text  = element_text(size = 11, face = "bold"),
      legend.title = element_text(size = 11, face = "bold"),
      legend.text  = element_text(size = 10, face = "bold"),
      legend.position = c(0.80, 0.25),
      legend.background = element_rect(fill = "transparent", color = NA),
      panel.border = element_rect(color = "black", linewidth = 1)
    )
  
  pdf(paste0(tt, "_Calibration.pdf"), 5.5, 5.5, family = "serif") 
  print(Calibrat_plot)
  dev.off()
}

# ============================================================
# 4.4 ROC, Best Threshold & Confusion Matrix
# ============================================================
get_best_threshold <- function(ROC) {
  out <- pROC::coords(ROC, x = "best", best.method = "youden", ret = c("threshold"), transpose = FALSE)
  if (is.data.frame(out) || is.matrix(out)) { 
    thr <- out[1, "threshold"] 
  } else if (is.list(out)) { 
    thr <- out[["threshold"]]
    if (length(thr) > 1) thr <- thr[1] 
  } else { 
    thr <- out 
  }
  as.numeric(thr)
}

plot_cm_gg <- function(ref, pred, title = NULL) {
  df <- as.data.frame(table(reference = ref, prediction = pred)) %>%
    dplyr::mutate(reference = factor(reference, levels = levels(ref)), prediction = factor(prediction, levels = levels(pred))) %>%
    dplyr::group_by(reference) %>% 
    dplyr::mutate(row_sum = sum(Freq), row_pct = ifelse(row_sum > 0, Freq / row_sum, NA_real_)) %>% 
    dplyr::ungroup()
  
  ggplot(df, aes(x = prediction, y = reference, fill = Freq)) +
    geom_tile(color = "grey85", linewidth = 0.4) +
    geom_text(aes(label = paste0(Freq, "\n(", sprintf("%.1f", 100 * row_pct), "%)")), size = 4, fontface = "bold", lineheight = 0.9) +
    scale_x_discrete(position = "top") + coord_equal() +
    labs(x = "Predicted", y = "Reference", title = title, fill = "Count") +
    theme_bw(base_family = "serif") + 
    theme(
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"), 
      panel.grid = element_blank(),
      axis.title = element_text(size = 12, face = "bold"),
      axis.text = element_text(size = 12, face = "bold")
    )
}

for (tt in names(datalist)) {
  newdata <- datalist[[tt]]
  ROC_list <- list()
  ROC_label <- list()
  Evaluation_metrics <- data.frame()
  
  for (mn in colnames(newdata)[-1]) {
    prob <- as.numeric(newdata[[mn]])
    
    if (all(is.na(prob)) || length(unique(prob[!is.na(prob)])) < 2) next
    
    ROC <- pROC::roc(response = newdata$outcome, predictor = prob, levels = c("improve", "fail"), direction = "<", quiet = TRUE)
    AUC <- as.numeric(pROC::auc(ROC))
    CI  <- as.numeric(pROC::ci.auc(ROC))
    bestp <- get_best_threshold(ROC)
    
    predlab <- factor(ifelse(prob > bestp, "fail", "improve"), levels = c("improve", "fail"))
    index_table <- caret::confusionMatrix(data = predlab, reference = newdata$outcome, positive = "fail", mode = "everything")
    
    pdf(paste0(tt, "_", mn, "_CM.pdf"), 5, 5, family = "serif")
    print(plot_cm_gg(ref = newdata$outcome, pred = predlab, title = paste0(tt, " - ", mn)))
    dev.off()
    
    Evaluation_metrics <- rbind(Evaluation_metrics, data.frame(
      Model = mn, Threshold = bestp, AUC = AUC, CI_low = CI[1], CI_high = CI[3],
      Accuracy = as.numeric(index_table$overall["Accuracy"]), 
      Sensitivity = as.numeric(index_table$byClass["Sensitivity"]),
      Specificity = as.numeric(index_table$byClass["Specificity"]), 
      Precision = as.numeric(index_table$byClass["Precision"]),
      F1 = as.numeric(index_table$byClass["F1"]),
      stringsAsFactors = FALSE
    ))
    
    ROC_label[[mn]] <- paste0(mn, " (AUC=", sprintf("%0.3f", AUC), ", 95% CI: ", sprintf("%0.3f", CI[1]), "-", sprintf("%0.3f", CI[3]), ")")
    ROC_list[[mn]] <- ROC
  }
  write.csv(Evaluation_metrics, paste0(tt, "_Evaluation_metrics.csv"), row.names = FALSE)
  
  roc_models <- names(ROC_list)
  cm <- get_color_map(roc_models)
  
  ROC_plot <- pROC::ggroc(ROC_list, size = 1.2, legacy.axes = TRUE) + 
    theme_bw(base_family = "serif") +
    labs(title = paste0(tt, " ROC curve")) +
    theme(
      plot.title = element_text(hjust = 0.5, size = 15, face = "bold"),
      axis.text = element_text(size = 12, face = "bold"),
      legend.title = element_blank(),
      legend.text = element_text(size = 10, face = "bold"),
      legend.position = c(0.70, 0.25),
      legend.background = element_blank(),
      axis.title = element_text(size = 12, face = "bold"),
      panel.border = element_rect(color="black", linewidth = 1), 
      panel.background = element_blank()
    ) +
    scale_colour_manual(values = cm, breaks = roc_models, labels = unname(ROC_label[roc_models])) +
    geom_segment(aes(x = 0, y = 0, xend = 1, yend = 1), colour = "grey", linetype = "dashed")
  
  pdf(paste0(tt, "_ROC.pdf"), 7, 7, family = "serif")
  print(ROC_plot)
  dev.off()
}

# ============================================================
# 4.5 Decision Curve Analysis (DCA)
# ============================================================
library(rmda)

for (tt in names(datalist)) {
  newdata <- datalist[[tt]]
  dca_data <- newdata
  dca_data$outcome <- ifelse(dca_data$outcome == "fail", 1, 0)
  DCA_list <- list()
  
  for (mn in colnames(dca_data)[-1]) {
    prob <- as.numeric(dca_data[[mn]])
    if (all(is.na(prob)) || length(unique(prob[!is.na(prob)])) < 2) next
    
    set.seed(123)
    DCA_list[[mn]] <- rmda::decision_curve(
      as.formula(paste("outcome ~", mn)), data = dca_data, family = binomial(link = "logit"),
      thresholds = seq(0.05, 0.95, by = 0.01), confidence.intervals = FALSE, study.design = "cohort"
    )
  }
  
  if (length(DCA_list) == 0) next
  dca_obj <- setNames(DCA_list, names(DCA_list))
  dca_models <- names(dca_obj)
  cm <- get_color_map(dca_models)
  col_vec <- unname(cm[dca_models])
  
  pdf(paste0(tt, "_DCA.pdf"), 7, 7, family = "serif")
  rmda::plot_decision_curve(
    dca_obj, curve.names = dca_models, col = col_vec, lty = rep(1, length(dca_models)),
    lwd = 2, confidence.intervals = FALSE, cost.benefit.axis = FALSE, 
    legend.position = "topright" 
  )
  title(main = paste0(tt, " Decision Curve Analysis"))
  dev.off()
}
# ============================================================
# 4.6 SHAP Analysis (Supports Manual or Automatic Model Selection)
# ============================================================

suppressPackageStartupMessages({
  library(kernelshap)
  library(shapviz)
  library(pROC)
  library(dplyr)
})

# ============================================================
# Step 0. Model Selection Strategy
# ============================================================

# Manually specify the model to explain:
best_Model <- "GBM"

# ============================================================
# Step 1. Automatic Selection (if best_Model is NULL)
# ============================================================

if (is.null(best_Model)) {
  
  test_auc_tbl <- data.frame(Model = character(), AUC = numeric())
  
  for (mn in colnames(datalist$Test)[-1]) {
    
    prob <- as.numeric(datalist$Test[[mn]])
    if (all(is.na(prob)) || length(unique(prob[!is.na(prob)])) < 2) next
    
    ROC <- pROC::roc(
      response  = datalist$Test$outcome,
      predictor = prob,
      levels    = c("improve","fail"),
      direction = "<",
      quiet     = TRUE
    )
    
    test_auc_tbl <- rbind(
      test_auc_tbl,
      data.frame(Model = mn, AUC = as.numeric(pROC::auc(ROC)))
    )
  }
  
  test_auc_tbl <- test_auc_tbl[order(-test_auc_tbl$AUC), ]
  write.csv(test_auc_tbl, "Test_AUC_ranking.csv", row.names = FALSE)
  
  best_Model <- test_auc_tbl$Model[1]
  
  cat(
    "\n[Auto-selected Best Model by Test AUC] => ",
    best_Model,
    " (AUC = ",
    round(test_auc_tbl$AUC[1], 3),
    ")\n",
    sep = ""
  )
  
} else {
  cat("\n[Manually Selected Model] => ", best_Model, "\n", sep = "")
}

# ============================================================
# Step 2. Prepare SHAP Input Data & Prediction Wrapper
# ============================================================

# Target data (X_dev): Explain the entire training set for stable global feature importance
X_dev <- Train_df %>% dplyr::select(-outcome)

# Background data (X_bg): Using the test set as the background distribution
X_bg  <- Test_df  %>% dplyr::select(-outcome)

# Unified prediction wrapper for kernelshap (returns 'fail' probabilities)
pred_fun_fail <- function(object, newdata) {
  nd <- as.data.frame(newdata)
  
  tmp <- data.frame(
    outcome = factor(rep("improve", nrow(nd)), levels = c("improve","fail")),
    nd
  )
  
  p <- get_prob_fail(
    model_obj   = object,
    model_name  = best_Model,
    newdata     = tmp,
    outcome_col = "outcome"
  )
  
  matrix(p, ncol = 1)
}

# Verify prediction function operates correctly
invisible(pred_fun_fail(models[[best_Model]], X_dev[1:5, ]))

# ============================================================
# Step 3. Compute SHAP Values
# ============================================================
set.seed(123)

shap_kernel <- kernelshap(
  models[[best_Model]],
  X        = X_dev,
  bg_X     = X_bg,
  pred_fun = pred_fun_fail
)

shap_value <- shapviz(
  shap_kernel,
  X = X_dev,
  interactions = TRUE
)

# ============================================================
# Step 4. Feature Direction Analysis (Correlation & Quartile)
# ============================================================

S_mat <- as.matrix(shap_value$S)  
X_mat <- as.matrix(shap_value$X) 

direction_df <- data.frame(
  Feature = colnames(X_mat),
  Spearman_Rho = NA_real_,
  Spearman_P = NA_real_,
  Mean_SHAP_Top25 = NA_real_,
  Mean_SHAP_Bottom25 = NA_real_,
  SHAP_Diff_T25_B25 = NA_real_,
  stringsAsFactors = FALSE
)

for (i in seq_along(direction_df$Feature)) {
  feat <- direction_df$Feature[i]
  
  # A) Spearman Correlation (SHAP > 0 increases fail probability)
  cor_test <- cor.test(X_mat[, feat], S_mat[, feat], method = "spearman", exact = FALSE)
  direction_df$Spearman_Rho[i] <- cor_test$estimate
  direction_df$Spearman_P[i]   <- cor_test$p.value
  
  # B) Top 25% vs Bottom 25% Mean SHAP Difference
  q_top <- quantile(X_mat[, feat], 0.75, na.rm = TRUE)
  q_bot <- quantile(X_mat[, feat], 0.25, na.rm = TRUE)
  
  idx_top <- which(X_mat[, feat] >= q_top)
  idx_bot <- which(X_mat[, feat] <= q_bot)
  
  m_top <- mean(S_mat[idx_top, feat], na.rm = TRUE)
  m_bot <- mean(S_mat[idx_bot, feat], na.rm = TRUE)
  
  direction_df$Mean_SHAP_Top25[i]    <- m_top
  direction_df$Mean_SHAP_Bottom25[i] <- m_bot
  direction_df$SHAP_Diff_T25_B25[i]  <- m_top - m_bot
}

direction_df <- direction_df %>%
  mutate(
    Direction = case_when(
      Spearman_Rho > 0 & SHAP_Diff_T25_B25 > 0 ~ "Positive (Risk Factor)",
      Spearman_Rho < 0 & SHAP_Diff_T25_B25 < 0 ~ "Negative (Protective)",
      TRUE ~ "Non-linear/Complex"
    )
  ) %>%
  arrange(desc(abs(Spearman_Rho)))

write.csv(direction_df, paste0("SHAP_", best_Model, "_direction_analysis.csv"), row.names = FALSE)

cat("\n[Success] Feature Direction Analysis Table Generated.\n")

# ============================================================
# Step 5. Output SHAP Visualizations (SCI Standard)
# ============================================================

# 1. Beeswarm Plot (Global Importance & Direction)
pdf(paste0("SHAP_", best_Model, "_beeswarm.pdf"), 7, 5, family = "serif")
sv_importance(
  shap_value,
  kind = "beeswarm",
  viridis_args = list(option = "B", begin = 0.25, end = 0.85),
  show_numbers = FALSE
) +
  theme_bw(base_family = "serif") +
  theme(axis.text = element_text(size = 12), axis.title = element_text(size = 14))
dev.off()

# 2. Bar Plot (Global Absolute Importance)
pdf(paste0("SHAP_", best_Model, "_bar.pdf"), 7, 5, family = "serif")
sv_importance(
  shap_value,
  kind = "bar",
  show_numbers = FALSE
) +
  theme_bw(base_family = "serif") +
  theme(axis.text = element_text(size = 12), axis.title = element_text(size = 14))
dev.off()

# 3. Dependence Plots (Clinical Validation)
target_features <- c("ALT", "TBil")

for (feat in target_features) {
  pdf(paste0("SHAP_", best_Model, "_dependence_", feat, ".pdf"), 5, 5, family = "serif")
  p <- sv_dependence(shap_value, v = feat) +
    theme_bw(base_family = "serif") +
    theme(axis.text = element_text(size = 11), axis.title = element_text(size = 13))
  print(p) 
  dev.off()
}

# 4. Waterfall Plots (Individual Case Analysis)
patient_fail_id    <- 81    
patient_improve_id <- 64
patient_fail_id    <- 37
# High-Risk Patient (Treatment Failure)
pdf(paste0("SHAP_", best_Model, "_waterfall_Fail_ID", patient_fail_id, ".pdf"), 5, 5, family = "serif")
p_fail <- sv_waterfall(shap_value, row_id = patient_fail_id) +
  ggtitle("High-Risk Patient (Treatment Failure)") +
  theme_bw(base_family = "serif") +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))
print(p_fail)
dev.off()

# Low-Risk Patient (Treatment Success)
pdf(paste0("SHAP_", best_Model, "_waterfall_Improve_ID", patient_improve_id, ".pdf"), 5, 5, family = "serif")
p_improve <- sv_waterfall(shap_value, row_id = patient_improve_id) +
  ggtitle("Low-Risk Patient (Treatment Success)") +
  theme_bw(base_family = "serif") +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))
print(p_improve)
dev.off()

cat("\n[Success] All customized SHAP plots saved for model: ", best_Model, "\n", sep = "")
