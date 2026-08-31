# ==============================================================================
# 1. Collinearity Diagnostics (VIF, Condition Index, Correlation Matrix)
# ==============================================================================
library(carData)
library(car)      
library(dplyr)    
library(corrplot) 
library(vcd)      
library(caret)
library(ggplot2)

# Load data
data <- read.csv("原始数据.csv", sep = ",", header = TRUE)

outcome_var <- names(data)[1]
continuous_vars <- names(data)[26:55]
categorical_vars <- names(data)[2:25]

outcome_data <- data[[outcome_var]]
continuous_data <- data[, continuous_vars]
categorical_data <- data[, categorical_vars]

# Handle missing values
all_data <- data.frame(
  outcome = outcome_data,
  continuous_data,
  categorical_data
)
names(all_data)[1] <- outcome_var
all_data <- na.omit(all_data)

outcome_clean <- all_data[[outcome_var]]
continuous_clean <- all_data[, continuous_vars]
categorical_clean <- all_data[, categorical_vars]

# ---------------------------------------------------------
# Handle categorical variables (convert to factors)
# ---------------------------------------------------------
categorical_factors <- data.frame(lapply(categorical_clean, function(x) {
  if (is.numeric(x)) {
    unique_vals <- unique(na.omit(x))
    if (all(unique_vals %in% c(0, 1))) {
      return(factor(x, levels = c(0, 1), labels = c("No", "Yes")))
    } else {
      return(x)
    }
  } else if (is.character(x) || !is.factor(x)) {
    return(factor(x))
  } else {
    return(x)
  }
}))
names(categorical_factors) <- names(categorical_clean)

# Create dummy variables (k-1 levels) to avoid perfect collinearity
create_safe_dummies <- function(factor_data) {
  dummy_list <- list()
  for (var_name in names(factor_data)) {
    var_data <- factor_data[[var_name]]
    
    if (is.factor(var_data)) {
      n_levels <- nlevels(var_data)
      levels_vec <- levels(var_data)
      
      if (n_levels == 2) {
        dummy_name <- var_name  
        dummy_list[[dummy_name]] <- as.numeric(var_data == levels_vec[2]) 
      } else if (n_levels > 2) {
        for (i in 2:n_levels) {
          clean_level <- gsub("[^[:alnum:]]", "_", levels_vec[i])
          dummy_name <- paste0(var_name, "_", clean_level)
          dummy_list[[dummy_name]] <- as.numeric(var_data == levels_vec[i])
        }
      }
    } else if (is.numeric(var_data)) {
      dummy_list[[var_name]] <- var_data
    }
  }
  return(as.data.frame(dummy_list))
}

categorical_dummies <- create_safe_dummies(categorical_factors)

continuous_predictors <- continuous_clean
if (outcome_var %in% names(continuous_predictors)) {
  continuous_predictors <- continuous_predictors[, names(continuous_predictors) != outcome_var]
}

# Merge predictors and remove perfectly duplicated columns
all_predictors <- cbind(continuous_predictors, categorical_dummies)
all_predictors <- all_predictors[, !duplicated(t(all_predictors))]

model_data <- data.frame(
  outcome = outcome_clean,
  all_predictors
)
names(model_data)[1] <- outcome_var

# Check for and remove linear combinations
X_data <- model_data[, -1]
linear_combos <- findLinearCombos(X_data)

if (!is.null(linear_combos$remove) && length(linear_combos$remove) > 0) {
  X_clean <- X_data[, -linear_combos$remove]
  model_data <- data.frame(outcome = outcome_clean, X_clean)
  names(model_data)[1] <- outcome_var
} 

# Build model for collinearity diagnostics
predictor_names <- names(model_data)[-1]
model_formula <- as.formula(paste(outcome_var, "~", paste(predictor_names, collapse = " + ")))
model <- lm(model_formula, data = model_data)

# ---------------------------------------------------------
# Variance Inflation Factor (VIF) Calculation
# ---------------------------------------------------------
tryCatch({
  vif_results <- vif(model)
  
  if (is.matrix(vif_results)) {
    if ("GVIF^(1/(2*Df))" %in% colnames(vif_results)) {
      vif_values <- vif_results[, "GVIF^(1/(2*Df))"]^2
    } else {
      vif_values <- vif_results[, 1]
    }
  } else {
    vif_values <- vif_results
  }
  
  vif_df <- data.frame(
    Variable = names(vif_values),
    VIF_Value = round(vif_values, 2),
    Status = ifelse(vif_values > 10, "Severe Collinearity",
                    ifelse(vif_values > 5, "Moderate Collinearity", "Acceptable")),
    stringsAsFactors = FALSE
  )
  vif_df <- vif_df[order(-vif_df$VIF_Value), ]
  
}, error = function(e) {
  cat("[Error] VIF calculation failed:", e$message, "\n")
})

# VIF Visualization
plot_df <- vif_df %>%
  arrange(VIF_Value) %>% 
  tail(30)             

plot_df$Variable <- factor(plot_df$Variable, levels = plot_df$Variable)

p_vif <- ggplot(plot_df, aes(x = Variable, y = VIF_Value, fill = Status)) +
  geom_bar(stat = "identity", width = 0.7, alpha = 0.8) +
  geom_hline(yintercept = 5, linetype = "dashed", color = "orange", size = 0.5) +
  geom_hline(yintercept = 10, linetype = "dashed", color = "red", size = 0.6) +
  coord_flip() +
  scale_fill_manual(values = c("Acceptable" = "#4575b4", 
                               "Moderate Collinearity" = "#fdae61", 
                               "Severe Collinearity" = "#d73027")) +
  theme_bw(base_size = 12) +
  theme(
    text = element_text(family = "serif"), 
    panel.grid.major.y = element_blank(),  
    legend.position = "bottom",            
    axis.title = element_text(face = "bold"),
    plot.title = element_text(hjust = 0.5, face = "bold")
  ) +
  labs(
    title = "Variance Inflation Factor (VIF) Diagnostic",
    x = "Variables",
    y = "VIF Value",
    fill = "Collinearity Status"
  )

ggsave("VIF_Diagnostic_SCI.pdf", plot = p_vif, width = 8, height = 10)


# ==============================================================================
# 2. Baseline Predictors Correlation Analysis
# ==============================================================================
vars_to_check <- c("ALT", "TBil", "INR", "Child_push_score") 

numeric_outcome <- ifelse(data$outcome == "deteriorate", 1, 0)

cor_results <- sapply(data[vars_to_check], function(x) {
  cor(x, numeric_outcome, use = "complete.obs", method = "pearson")
})

print("Correlation coefficients between baseline predictors and binary outcome:")
print(cor_results)

cor_alt_test <- cor.test(data$ALT, numeric_outcome, method = "pearson")
print(cor_alt_test)


# ==============================================================================
# 3. Outcome-Component Analysis
# ==============================================================================
df_failed <- data_new %>% 
  filter(type %in% c(1, 2, 3)) %>%
  mutate(Event_Category = case_when(
    type %in% c(1, 3) ~ "Hard Events (N=31)",
    type == 2 ~ "Biochemical Only (N=146)",
    TRUE ~ "Other"
  ))

df_failed$Event_Category <- factor(df_failed$Event_Category, 
                                   levels = c("Biochemical Only (N=146)", 
                                              "Hard Events (N=31)"))

prob_bio <- df_failed$Predicted_Prob[df_failed$Event_Category == "Biochemical Only (N=146)"]
prob_hard <- df_failed$Predicted_Prob[df_failed$Event_Category == "Hard Events (N=31)"]

# Wilcoxon rank-sum test
wilcox_res <- wilcox.test(Predicted_Prob ~ Event_Category, data = df_failed)
p_val <- wilcox_res$p.value

# Cohen's d
n1 <- length(prob_bio)
n2 <- length(prob_hard)
v1 <- var(prob_bio, na.rm=TRUE)
v2 <- var(prob_hard, na.rm=TRUE)
pooled_sd <- sqrt(((n1-1)*v1 + (n2-1)*v2) / (n1+n2-2))
cohens_d <- abs(mean(prob_hard, na.rm=TRUE) - mean(prob_bio, na.rm=TRUE)) / pooled_sd

# Kolmogorov-Smirnov (KS) test
ks_res <- ks.test(prob_bio, prob_hard)
p_val_ks <- ks_res$p.value

p_label <- ifelse(p_val < 0.001, "P < 0.001", sprintf("P = %.3f", p_val))

# Visualization
ggplot(df_failed, aes(x = Event_Category, y = Predicted_Prob, fill = Event_Category)) +
  geom_violin(alpha = 0.3, trim = FALSE, color = NA) +
  geom_boxplot(width = 0.2, alpha = 0.9, outlier.shape = NA, color = "black") +
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


# ==============================================================================
# 4. GBM-Specific Stratified SHAP Dependence Analysis
# ==============================================================================
vars_needed <- c("ALT", "Cirrhosis")
complete_idx <- complete.cases(traindata[, vars_needed])

analysis_df <- traindata[complete_idx, ]
analysis_df$Cirrhosis <- as.numeric(as.character(analysis_df$Cirrhosis))

shap_ALT <- shap_value$X[complete_idx, "ALT"]
shap_val <- shap_value$S[complete_idx, "ALT"]
filtered_cirrhosis <- analysis_df$Cirrhosis

plot_shap_df <- data.frame(
  ALT = shap_ALT,
  SHAP = shap_val,
  Cirrhosis_Label = factor(filtered_cirrhosis, levels = c(0, 1), labels = c("Non-Cirrhotic", "Cirrhotic"))
)

p_shap <- ggplot(plot_shap_df, aes(x = ALT, y = SHAP, color = Cirrhosis_Label)) +
  geom_point(alpha = 0.7, size = 2.2) +
  geom_smooth(method = "loess", se = TRUE, linewidth = 1) + 
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
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
    panel.grid.minor = element_blank()
  )

ggsave("Supplementary_Fig_4B_Stratified_SHAP.pdf", plot = p_shap, width = 7, height = 5, dpi = 300)


# ==============================================================================
# 5. Cohort Development: Imputation, Train/Test Split, and Baseline Tables
# ==============================================================================
library(naniar)
library(mice) 
library(tableone)

data = read.csv("Duli最新.csv", header = TRUE, encoding = "GBK")
data[,1:24] <- lapply(data[,1:24], as.factor) 
continuous_vars <- 25:ncol(data)

# Missing value visualization and export
missing_ratio <- colMeans(is.na(data))
missing_ratio_df <- data.frame(Variable = names(missing_ratio), Missing_Ratio = missing_ratio)
write.csv(missing_ratio_df, "Missing_Value_Ratio.csv", row.names = FALSE)

data_imputed <- mice(data, seed = 123, print = FALSE, m = 5)
data_imputed2 <- mice::complete(data_imputed, action = 2)
write.csv(data_imputed2, "data_complete.csv", row.names = FALSE)

# Read imputed data and prepare dataset
data = read.csv("data_complete.csv", header = TRUE, encoding = "GBK")
data$outcome <- factor(data$outcome, levels = c(0, 1), labels = c("Treatment Success", "Treatment Failure"))

set.seed(123) 
inTrain = createDataPartition(y=data[,"outcome"], p=0.7, list=FALSE) 
traindata = data[inTrain,] 
testdata = data[-inTrain,] 
write.csv(traindata, "traindata.csv", row.names = FALSE) 
write.csv(testdata, "testdata.csv", row.names = FALSE) 

# Baseline Table Generation
myVars <- colnames(data[, 2:ncol(data)])  
catVars <- colnames(data[, 2:25])         
contVars <- colnames(data[, 26:ncol(data)])  

noncenter <- c()
for (var in contVars) {
  if (is.numeric(data[[var]])) {
    test_result <- shapiro.test(data[[var]])
    if (test_result$p.value < 0.05) {
      noncenter <- c(noncenter, var)
    }
  }
}

table1 <- CreateTableOne(vars = myVars, factorVars = catVars, data = data, strata='outcome', addOverall = TRUE)  
table1 <- print(table1, showAllLevels = TRUE, catDigits = 2, contDigits = 2, pDigits = 3, nonnormal = noncenter)
write.csv(table1, file = "Overall_Baseline_Table.csv") 

table2 <- CreateTableOne(vars = myVars, factorVars = catVars, data = traindata, strata='outcome', addOverall = TRUE)  
table2 <- print(table2, showAllLevels = TRUE, catDigits = 2, contDigits = 2, pDigits = 3, nonnormal = noncenter)
write.csv(table2, file = "Training_Baseline_Table.csv")


# ==============================================================================
# 6. Single Feature ROC Analysis (e.g., Child-Pugh score)
# ==============================================================================
library(pROC)

data <- read.csv("testdata.csv", header = TRUE)
data$outcome <- factor(data$outcome, levels = c(0, 1), labels = c("Success", "Failure"))

roc_data <- data.frame(
  Child_push_score = data$Child_push_score,
  Outcome = data$outcome
)

tryCatch({
  roc_obj <- roc(Outcome ~ Child_push_score, data = roc_data, 
                 levels = c("Success", "Failure"), 
                 direction = "<") 
  
  auc_value <- auc(roc_obj)
  ci_value <- ci(roc_obj, of = "auc")
  
  auc_text <- sprintf("%.3f", auc_value)
  ci_text <- sprintf("%.3f - %.3f", ci_value[1], ci_value[3])
  
  plot(roc_obj, 
       main = paste0("ROC Curve for Child-Pugh Score\nAUC = ", auc_text, " (95% CI: ", ci_text, ")"),
       col = "#2E86AB", 
       lwd = 2,
       legacy.axes = TRUE,
       print.auc = FALSE, 
       grid = TRUE)
  
  abline(a = 0, b = 1, lty = 2, col = "red")
  saveRDS(roc_obj, "ROC_object_ChildPugh.rds")
  
}, error = function(e) {
  cat("[Error] ROC calculation failed:", e$message, "\n")
})