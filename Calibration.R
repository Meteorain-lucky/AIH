# ============================================================
# 4.3 Calibration Curve and Quantitative Metrics (Multi-model)
# Note: caret::calibration expects formula: outcome ~ model1 + model2 + ...
# ============================================================

cutpoint <- 5  

# Store quantitative calibration metrics across datasets
all_calib_metrics <- list()

for (tt in names(datalist)) {
  
  newdata <- datalist[[tt]]
  
  # Ensure outcome is a factor and contains the "fail" level
  stopifnot("outcome" %in% colnames(newdata))
  stopifnot("fail" %in% levels(newdata$outcome))
  
  # ------------------------------------------------------------
  # Step 1. Calculate quantitative calibration metrics 
  # (Brier Score, Calibration Intercept, Slope, HL test)
  # ------------------------------------------------------------
  model_names <- colnames(newdata)[-1]
  metrics_list <- list()
  
  # Convert outcome to binary numeric (fail = 1, others = 0)
  y_bin <- ifelse(newdata$outcome == "fail", 1, 0)
  
  for (m in model_names) {
    p_pred <- newdata[[m]] 
    
    # Calculate Brier Score
    brier_score <- mean((p_pred - y_bin)^2, na.rm = TRUE)
    
    # Restrict probabilities to avoid exactly 0 or 1, preventing glm fitting errors
    p_pred_safe <- pmin(pmax(p_pred, 1e-15), 1 - 1e-15)
    logit_pred <- log(p_pred_safe / (1 - p_pred_safe))
    
    # Calibration Intercept: Fit logit(p) as an offset
    fit_intercept <- tryCatch({
      glm(y_bin ~ 1, offset = logit_pred, family = binomial())
    }, error = function(e) NULL)
    cal_intercept <- if(!is.null(fit_intercept)) coef(fit_intercept)[1] else NA_real_
    
    # Calibration Slope: Fit logit(p) as the predictor
    fit_slope <- tryCatch({
      glm(y_bin ~ logit_pred, family = binomial())
    }, error = function(e) NULL)
    cal_slope <- if(!is.null(fit_slope)) coef(fit_slope)[2] else NA_real_
    
    # Hosmer-Lemeshow Test via ResourceSelection package
    hl_pval <- tryCatch({
      if (requireNamespace("ResourceSelection", quietly = TRUE)) {
        hl <- ResourceSelection::hoslem.test(y_bin, p_pred, g = min(10, length(y_bin)))
        hl$p.value
      } else {
        NA_real_
      }
    }, error = function(e) NA_real_)
    
    metrics_list[[m]] <- data.frame(
      Dataset = tt,
      Model = m,
      Brier_Score = brier_score,
      Calibration_Intercept = cal_intercept,
      Calibration_Slope = cal_slope,
      HL_P_Value = hl_pval
    )
  }
  
  # Combine metrics for all models in the current dataset
  all_calib_metrics[[tt]] <- do.call(rbind, metrics_list)
  
  # ------------------------------------------------------------
  # Step 2. Plot calibration curves
  # ------------------------------------------------------------
  formula_cal <- as.formula(
    paste0("outcome ~ ", paste(model_order <- colnames(newdata)[-1], collapse = " + "))
  )
  
  # Generate calibration object
  cal_obj <- caret::calibration(formula_cal, data = newdata, class = "fail", cuts = cutpoint)
  caldata <- as.data.frame(cal_obj$data) %>% tidyr::drop_na()
  
  # Ensure model factor levels match column order for consistency
  model_order <- colnames(newdata)[-1]
  caldata$calibModelVar <- factor(caldata$calibModelVar, levels = model_order)
  
  # Map colors to models
  cm <- get_color_map(model_order)
  
  Calibrat_plot <- ggplot(
    caldata,
    aes(x = midpoint, y = Percent, group = calibModelVar, color = calibModelVar)
  ) +
    geom_point(size = 1.6) +
    geom_line(linewidth = 0.8) +
    geom_abline(slope = 1, intercept = 0, color = "black", linetype = "dashed") +
    scale_color_manual(values = cm, name = "Model", drop = FALSE) +
    labs(
      title = paste0(tt, " Calibration"),
      x = "Predicted risk (bin midpoint)",
      y = "Observed event rate"
    ) +
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

# ------------------------------------------------------------
# Step 3. Export quantitative results to CSV
# ------------------------------------------------------------
final_metrics_df <- do.call(rbind, all_calib_metrics)

# Format metrics (now columns 3 to 6) to 3 decimal places
final_metrics_df[, 3:6] <- lapply(final_metrics_df[, 3:6], function(x) sprintf("%.3f", as.numeric(x)))

write.csv(final_metrics_df, "Quantitative_Calibration_Metrics.csv", row.names = FALSE)
cat("\n[Success] Train/Test calibration metrics saved to: Quantitative_Calibration_Metrics.csv\n")


# ============================================================
# 5.1 Test2 Calibration Curve + Quantitative Metrics
# ============================================================
cat("\nPlotting Test2 Calibration Curve and calculating metrics...\n")

# Store quantitative calibration results for the independent validation cohort
test2_calib_metrics <- list() 

for (tt in names(datalist_test2)) {
  newdata <- datalist_test2[[tt]]
  cutpoint <- 5  
  
  stopifnot("outcome" %in% colnames(newdata))
  stopifnot("fail" %in% levels(newdata$outcome))
  
  model_order <- colnames(newdata)[-1]
  
  # ------------------------------------------------------------
  # Step 1. Calculate quantitative calibration metrics (Test2)
  # ------------------------------------------------------------
  metrics_list <- list()
  y_bin <- ifelse(newdata$outcome == "fail", 1, 0)
  
  for (m in model_order) {
    p_pred <- newdata[[m]] 
    
    # Calculate Brier Score
    brier_score <- mean((p_pred - y_bin)^2, na.rm = TRUE)
    
    p_pred_safe <- pmin(pmax(p_pred, 1e-15), 1 - 1e-15)
    logit_pred <- log(p_pred_safe / (1 - p_pred_safe))
    
    # Calibration Intercept
    fit_intercept <- tryCatch({
      glm(y_bin ~ 1, offset = logit_pred, family = binomial())
    }, error = function(e) NULL)
    cal_intercept <- if(!is.null(fit_intercept)) coef(fit_intercept)[1] else NA_real_
    
    # Calibration Slope
    fit_slope <- tryCatch({
      glm(y_bin ~ logit_pred, family = binomial())
    }, error = function(e) NULL)
    cal_slope <- if(!is.null(fit_slope)) coef(fit_slope)[2] else NA_real_
    
    # Hosmer-Lemeshow Test
    hl_pval <- tryCatch({
      if (requireNamespace("ResourceSelection", quietly = TRUE)) {
        hl <- ResourceSelection::hoslem.test(y_bin, p_pred, g = min(10, length(y_bin)))
        hl$p.value
      } else {
        NA_real_
      }
    }, error = function(e) NA_real_)
    
    metrics_list[[m]] <- data.frame(
      Dataset = tt,
      Model = m,
      Brier_Score = brier_score,
      Calibration_Intercept = cal_intercept,
      Calibration_Slope = cal_slope,
      HL_P_Value = hl_pval
    )
  }
  
  test2_calib_metrics[[tt]] <- do.call(rbind, metrics_list)
  
  # ------------------------------------------------------------
  # Step 2. Plot calibration curves (Test2)
  # ------------------------------------------------------------
  formula_cal <- as.formula(paste0("outcome ~ ", paste(model_order, collapse = " + ")))
  cal_obj <- caret::calibration(formula_cal, data = newdata, class = "fail", cuts = cutpoint)
  caldata <- as.data.frame(cal_obj$data) %>% tidyr::drop_na()
  
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

# ------------------------------------------------------------
# Step 3. Export quantitative results to CSV (Test2)
# ------------------------------------------------------------
final_test2_metrics_df <- do.call(rbind, test2_calib_metrics)

# Format metrics (now columns 3 to 6) to 3 decimal places
final_test2_metrics_df[, 3:6] <- lapply(final_test2_metrics_df[, 3:6], function(x) sprintf("%.3f", as.numeric(x)))

write.csv(final_test2_metrics_df, "Quantitative_Calibration_Metrics_Test2.csv", row.names = FALSE)
cat("\n[Success] Test2 quantitative calibration metrics saved to: Quantitative_Calibration_Metrics_Test2.csv\n")

