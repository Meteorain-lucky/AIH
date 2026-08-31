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

