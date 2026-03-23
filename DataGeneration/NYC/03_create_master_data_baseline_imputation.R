######################## Fall 2024: Master Data Set ############################
# AIM:
# Create master data set:
# Ranking by residential district 

# DATA SOURCE:

# 1) Government Reports:

# Student Applications, Admissions and Offers

# 1.1) Applications & Offers: Report on Available Seats, Application and Offers 2024-2025
# https://infohub.nyced.org/reports/government-reports/student-applications-admissions-and-offers
# All available seats, applicants, and offers of admissions are for the main round of
# admissions only. Data is only included for schools that participate in the centralized admissions processes.
# This data set is relevant as it includes the columns "Total" and "True" applicants/ applicants
# that did not receive an offer to a more preferred school.

# Packages 
library(readxl)
library(dplyr)
library(writexl)
library(stringr)
library(withr)
library(tibble)
library(purrr)
library(readr)

# Load data --------------------------------------------------------------------

# Set path to data folder
# The folder should contain the downloaded data sets
data_path <- Sys.getenv("DATA_PATH_NYC")

# Load functions
source("DataGeneration/NYC/src/convert_s.R")
source("DataGeneration/NYC/src/impute_s.R")
source("DataGeneration/NYC/src/run_one_seed.R")
source("DataGeneration/NYC/src/postprocess_master_data.R")


# Load data sets

## Applications
# Rows: 137.392
# Col: 24
applications_by_school <- read_excel(paste0(data_path,"/fall-2024-admissions-72-suppressed.xlsx"), 
                                     sheet = "School")

## Sum of applications by school (from 02_create_school_applicants)
# Rows: 422
# Col: 10
school_applicants_02 <- read_excel("DataGeneration/NYC/outputs/02_school_applicants.xlsx")
school_applicants_02 <- school_applicants_02 %>%
  select(`School DBN`, `Grade 9 Total Applicants`, `Grade 9 True Applicants`) %>%
  rename(
    Total_Applicants = `Grade 9 Total Applicants`,
    True_Applicants  = `Grade 9 True Applicants`)

## Prep ------------------------------------------------------------------------

# Keep relevant columns
applications_by_school <- applications_by_school %>%
  select(`School District`, `School DBN`, `School Name`,
         `Category`,`Grade 9 Total Applicants`, `Grade 9 True Applicants`)


# Convert s to "-1" and s^ to "-2" 
applications_by_school <- applications_by_school %>%
  mutate(
    `Grade 9 Total Applicants`   = convert_col(`Grade 9 Total Applicants`),
    `Grade 9 True Applicants`            = convert_col(`Grade 9 True Applicants`)
  )

# Remove NA
# Rows: 57.050
applications_by_school <- applications_by_school %>%
  filter(!is.na(`Grade 9 Total Applicants` ) | !is.na(`Grade 9 True Applicants`))


# Remove Specialized HS 
# Rows: 54.327
specialized_HS <- c("13K430","02M475","05M692", "10X445", "14K449", "28Q687", "10X696", "31R605", "03M485")
applications_by_school <- applications_by_school %>%
  filter(!`School DBN` %in% specialized_HS)

# Subset
# Rows: 10.363
applications_by_school_residential_district <- applications_by_school %>%
  filter(grepl("^Residential District", Category)) %>%
  mutate(Category = str_remove(Category, "^Residential District\\s+"))
rm(applications_by_school)

# Convert
# "Unknown" to 100 and to integer
applications_by_school_residential_district <- applications_by_school_residential_district %>%
  mutate(Category = if_else(Category == "Unknown", "100", Category))%>%
  mutate(Category = as.integer(Category)) 

# Remove 
# Schools with 
# 1) Only "s" or "s^" values in true | total
# 2) at least on "s^" value in true | total --> this means: "s^" is not imputed

# Rows: 9.979
# 403 schools
applications_by_school_residential_district <- applications_by_school_residential_district %>%
  group_by(`School DBN`) %>%
  filter(
    !(all(`Grade 9 Total Applicants` %in% c(-1, -2)) |
        all(`Grade 9 True Applicants`  %in% c(-1, -2))
    )
  ) %>%
  filter(
    !(any(`Grade 9 Total Applicants` == -2) |
        any(`Grade 9 True Applicants`  == -2)
    )
  ) %>%
  ungroup()

# Join with total number of studenty by school
applications_by_school_residential_district <- applications_by_school_residential_district %>%
  left_join(school_applicants_02, by = "School DBN")
rm(school_applicants_02)

## Impute  ---------------------------------------------------------------------
# Impute "Total" & "True"

# Create 100 random seeds
set.seed(123)         
seeds <- sample.int(1e6, 100)

# Run imputations for each seed
runs <- map(
  seeds,
  ~ run_one_seed(applications_by_school_residential_district, .x)
)

# Validate imputation results
validation_table <- map_dfr(runs, "diagnostics")
validation_table %>% count(passed)

# Extract imputed data sets
imputed_datasets <- map(runs, "data")

# Postprocess
imputed_datasets_post <- map(runs, "data") %>%
  map(postprocess_seed_dataset)


# Save -------------------------------------------------------------------------
out_dir <- "DataGeneration/NYC/outputs/data_baseline_imputation"
csv_dir <- file.path(out_dir, "csv")

walk2(
  imputed_datasets_post,
  seeds,
  ~ readr::write_csv(
    .x,
    file = file.path(csv_dir, paste0("imputed_seed_", .y, ".csv"))
  )
)

zip(
  zipfile = file.path(out_dir, "data_baseline_imputation_seeds_1_to_100.zip"),
  files   = list.files(csv_dir, full.names = TRUE)
)

























