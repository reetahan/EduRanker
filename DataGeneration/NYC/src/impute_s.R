# Aim
# Impute s with 0-5 and s^ with >5

## Total Applicants ------------------------------------------------------------

# The function randomly splits a total (remaining sum of total applicants by school)
# across n slots (district, zip codes, home languages) such that each slot
# ends up having a value between 0-5
# E.g.: random_allocate(n = 4, total = 7, min_val = 0, max_val = 5)
# gives 3 2 2 0, or 3 2 1 1, etc.

random_allocate <- function(n, total, min_val = 0, max_val = 5) {
  if (n == 0) return(integer(0))
  
  if (total < n * min_val || total > n * max_val) {
    stop("Allocation impossible with bounds 0–5")
  }
  
  vals <- rep(min_val, n)
  remaining <- total
  
  while (remaining > 0) {
    i <- sample(seq_len(n), 1)
    if (vals[i] < max_val) {
      vals[i] <- vals[i] + 1
      remaining <- remaining - 1
    }
  }
  
  vals
}


# The function imputes missing values "s" (-1) in the "Grade 9 Total Applicants" by school (!!!)
# If the sum of "Grade 9 Total Applicants" by school = "Total Applicants" by school,
# than no imputation is done and same values are inserted in the new column "imputed",
# otherwise if fills the values with random values using the function above

impute_total_school <- function(df_school) {
  
  total_school <- df_school$Total_Applicants[1]
  g9 <- df_school$`Grade 9 Total Applicants`
  
  # Create new column "imputed"
  df_school$imputed <- g9
  
  observed_sum <- sum(g9[g9 != -1], na.rm = TRUE)
  
  # If sum equal to total sum by school --> keep original values or replace -1 with 0
  if (observed_sum == total_school) {
    df_school$imputed[g9 == -1] <- 0L
    return(df_school)
  }
  
  idx_missing <- which(g9 == -1)
  n_missing <- length(idx_missing)
  
  if (n_missing == 0) {
    return(df_school)
  }
  
  remaining <- total_school - observed_sum # sum that needs to be allocated
  
  # If the sum is achieved, and the remaining values are "-1" impute with 0
  if (remaining <= 0) {
    df_school$imputed[idx_missing] <- 0L
    return(df_school)
  }
  
  imputed_vals <- random_allocate(
    n = n_missing,
    total = remaining,
    min_val = 0,
    max_val = 5
  )
  
  df_school$imputed[idx_missing] <- imputed_vals
  
  df_school
}

# True Applicants --------------------------------------------------------------

# The function randomly splits a total (remaining sum of true applicants by school)
# across n slots (district, zip codes, home languages) but
# each slot has its own upper bound (given that true applicants ≤ total applicants by slot)
# E.g.: random_allocate_bounded(upper_bounds = c(2, 5, 1), total = 5, min_val = 0, max_val = 5)
# gives 1 3 1, or 2 2 1 etc.

random_allocate_bounded <- function(upper_bounds, total, min_val = 0, max_val = 5) {
  
  n <- length(upper_bounds)
  if (n == 0) return(integer(0))
  
  # enforce global max_val in addition to row-specific bounds
  effective_bounds <- pmin(upper_bounds, max_val)
  
  max_possible <- sum(effective_bounds)
  if (total < n * min_val || total > max_possible) {
    stop("Allocation impossible given bounds")
  }
  
  vals <- rep(min_val, n)
  remaining <- total
  
  while (remaining > 0) {
    i <- sample(seq_len(n), 1)
    if (vals[i] < effective_bounds[i]) {
      vals[i] <- vals[i] + 1
      remaining <- remaining - 1
    }
  }
  
  vals
}


# The function imputes missing values "s" (-1) in the "Grade 9 True Applicants" by school (!!!)
# If the sum of "Grade 9 True Applicants" by school = "True Applicants" by school,
# than no imputation is done and same values are inserted in the new column "True_Applicants_Imputed",
# otherwise if fills the values with random values using the function above

impute_true_school <- function(df_school) {
  
  true_school_total <- df_school$True_Applicants[1]
  true_vals <- df_school$`Grade 9 True Applicants`
  
  # Create new column "True_Applicants_Imputed"
  df_school$True_Applicants_Imputed <- true_vals
  
  observed_sum <- sum(true_vals[true_vals != -1], na.rm = TRUE)
  
  # If already consistent → keep original or replace -1 with 0
  if (observed_sum == true_school_total) {
    df_school$True_Applicants_Imputed[true_vals == -1] <- 0L
    return(df_school)
  }
  
  idx_missing <- which(true_vals == -1)
  n_missing <- length(idx_missing)
  
  if (n_missing == 0) {
    return(df_school)
  }
  
  remaining <- true_school_total - observed_sum
  
  # If the sum is achieved, and the remaining values are "-1" impute with 0
  if (remaining <= 0) {
    df_school$True_Applicants_Imputed[idx_missing] <- 0L
    return(df_school)
  }
  
  # upper bounds: must be ≤ imputed total applicants AND ≤ 5
  upper_bounds <- pmin(df_school$imputed[idx_missing], 5)
  
  imputed_vals <- random_allocate_bounded(
    upper_bounds = upper_bounds,
    total = remaining,
    min_val = 0
  )
  
  df_school$True_Applicants_Imputed[idx_missing] <- imputed_vals
  
  df_school
}










