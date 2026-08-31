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
data <- read.csv("data_complete.csv", sep = ",", header = TRUE)
###data <- read.csv("data_complete new.csv", sep = ",", header = TRUE)
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


