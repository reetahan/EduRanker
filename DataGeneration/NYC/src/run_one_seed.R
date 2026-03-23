run_one_seed <- function(dat, seed) {
  
  imputed <- withr::with_seed(seed, {
    dat %>%
      group_by(`School DBN`) %>%
      group_modify(~ impute_total_school(.x)) %>%
      ungroup() %>%
      group_by(`School DBN`) %>%
      group_modify(~ impute_true_school(.x)) %>%
      ungroup()
  })
  
  # Validate
  
  # Sum for total applicants by school correct?
  mismatch_total <- imputed %>%
    group_by(`School DBN`) %>%
    summarise(
      sum_by_res = sum(imputed, na.rm = TRUE),
      total_school = first(Total_Applicants),
      .groups = "drop"
    ) %>%
    filter(sum_by_res != total_school)
  
  # Sum for true applicants by school correct?
  mismatch_true <- imputed %>%
    group_by(`School DBN`) %>%
    summarise(
      sum_by_res = sum(True_Applicants_Imputed, na.rm = TRUE),
      total_school = first(True_Applicants),
      .groups = "drop"
    ) %>%
    filter(sum_by_res != total_school)
  
 # Total applications >= true applications
  violations <- imputed %>%
    filter(imputed < True_Applicants_Imputed)
  
  diag <- tibble(
    seed = seed,
    n_mismatch_total = nrow(mismatch_total),
    n_mismatch_true  = nrow(mismatch_true),
    n_violations     = nrow(violations),
    passed = (n_mismatch_total == 0 &&
                n_mismatch_true  == 0 &&
                n_violations     == 0)
  )
  
  list(
    seed = seed,
    data = imputed,
    diagnostics = diag
  )
}