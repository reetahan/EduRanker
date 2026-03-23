postprocess_seed_dataset <- function(df) {
  
  df %>%
    
    # Rename & reorder
    select(
      `School DBN`,
      `School Name`,
      `School District`,
      `Residential District` = Category,
      `Total Applicants by Residential District` = imputed,
      `Total Applicants by Residential District Not Imputed` = `Grade 9 Total Applicants`,
      `True Applicants by Residential District` = True_Applicants_Imputed,
      `True Applicants by Residential District Not Imputed` = `Grade 9 True Applicants`,
      `Total Applicants School` = Total_Applicants,
      `Total True Applicants School` = True_Applicants
    ) %>%
    
    # Convert Residential District = 100 to "Unknown"
    mutate(
      `Residential District` = if_else(
        `Residential District` == 100,
        "Unknown",
        as.character(`Residential District`)
      )
    ) %>%
    
    # Remove rows where total or true is 0
    filter(
      `Total Applicants by Residential District` != 0,
      `True Applicants by Residential District` != 0
    ) %>%
    
    # Create ratio
    mutate(
      Ratio = (`True Applicants by Residential District`^2) /
        (`Total Applicants by Residential District`)
    ) %>%
    
    # Rank within residential district
    group_by(`Residential District`) %>%
    mutate(Rank = dense_rank(desc(Ratio))) %>%
    ungroup() %>%
    arrange(`Residential District`, Rank)
}
