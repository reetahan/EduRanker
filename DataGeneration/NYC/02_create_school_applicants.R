######################## Fall 2024: School Applicants ###########################
# AIM:
# Create data set
# (1): Applications by school: Total & True applicants 

# DATA SOURCES:

# 1) Government Reports:

# Student Applications, Admissions and Offers

# 1.1) Applications & Offers: Report on Available Seats, Application and Offers 2024-2025
# https://infohub.nyced.org/reports/government-reports/student-applications-admissions-and-offers
# All available seats, applicants, and offers of admissions are for the main round of
# admissions only. Data is only included for schools that participate in the centralized admissions processes.
# This data set is relevant as it includes the columns "Total" and "True" applicants/ applicants
# that did not receive an offer to a more preferred school.

# 1.2) Enrollments: Report on Enrollment 2024-2025
# https://infohub.nyced.org/reports/government-reports/student-applications-admissions-and-offers
# Number of enrollments by school and category 

# 2) Admissions and Enrollment:

# 2.1) School Directory Data: 
# https://infohub.nyced.org/reports/admissions-and-enrollment/directory-data
# It includes information about high schools e.g.neighborhood, program highlights as
# well as information on the number of applicants & seats (specialized HS, general education HS, disabilities)
# by program

# 2.2)  Matches to Choice Data:
# https://infohub.nyced.org/reports/admissions-and-enrollment/admissions-outcomes
# % of students matched to one of their top choices by residential district and ethnicity/race group

# Packages 
library(readxl)
library(dplyr)
library(writexl)
library(stringr)

# Load data --------------------------------------------------------------------

# Set path to data folder
# The folder should contain the downloaded data sets
data_path <- Sys.getenv("DATA_PATH_NYC")

# Load functions
source("DataGeneration/NYC/src/convert_s.R")

# Load data sets

## 1.1) Applications
# Rows: 137.392
# Col: 24
applications_by_school <- read_excel(paste0(data_path,"/fall-2024-admissions-72-suppressed.xlsx"), 
                                     sheet = "School")

## 1.2) Enrollments
# Rows: 78.105
# Col: 9
enrollments_by_school <- read_excel(paste0(data_path,"/fall-2024-admissions_part-ii_suppressed.xlsx"), 
                                    sheet = "School")

## 2.1) School Directory Data
# Rows: 452
# Col: 371
directory_data_by_school <- read_excel(paste0(data_path,"/fall-2025---hs-directory-data.xlsx"), 
                                       sheet = "Data")


# DATA 02 ----------------------------------------------------------------
# Applications by school data set

## Prepare ---------------------------------------------------------------------

## 1.1)

# Keep relevant columns
applications_by_school_02 <- applications_by_school %>%
  select(`School District`, `School DBN`, `School Name`,
         `Category`,`Grade 9 Total Applicants`, `Grade 9 True Applicants`)


# Convert s to -1 and s^ to -2 
applications_by_school_02 <- applications_by_school_02 %>%
  mutate(
    `Grade 9 Total Applicants`   = convert_col(`Grade 9 Total Applicants`),
    `Grade 9 True Applicants`            = convert_col(`Grade 9 True Applicants`)
  )

# Remove NA
# (Also maybe if -2)
# Rows: 57050 (54327)
applications_by_school_02 <- applications_by_school_02 %>%
  filter(!is.na(`Grade 9 Total Applicants` ) | !is.na(`Grade 9 True Applicants`))


# Create total data set 
# Rows: 431
applications_by_school_02 <- applications_by_school_02  %>%
  filter(Category == "All Students") %>%
  select(-Category) 

## 2.1)

# Keep relevant columns
directory_data_by_school_02 <- directory_data_by_school %>%
  select( dbn, applicants1specialized, applicants2specialized,
          applicants3specialized, applicants4specialized, 
          applicants5specialized, applicants6specialized,
          grade9geapplicants1, grade9geapplicants2,
          grade9geapplicants3, grade9geapplicants4,
          grade9geapplicants5, grade9geapplicants6,
          grade9geapplicants7, grade9geapplicants8,
          grade9geapplicants9, grade9geapplicants10,
          grade9geapplicants11,
          grade9swdapplicants1, grade9swdapplicants2,
          grade9swdapplicants3, grade9swdapplicants4,
          grade9swdapplicants5, grade9swdapplicants6,
          grade9swdapplicants7, grade9swdapplicants8,
          grade9swdapplicants9, grade9swdapplicants10,
          grade9swdapplicants11)

# Remove NAs
# Rows: 444
directory_data_by_school_02 <- directory_data_by_school_02 %>%
  filter(!if_all(-dbn, ~ is.na(.x)))

# Convert columns to numeric
directory_data_by_school_02 <- directory_data_by_school_02 %>%
  mutate(across(-dbn, ~ suppressWarnings(as.numeric(.))))

# Create column
# Create a column that sums the number of applicants of
# Seats at Specialized HS, General Education HS,
# and Students with Disabilities HS
directory_data_by_school_02 <- directory_data_by_school_02 %>%
  rowwise() %>%
  mutate(
    `Total True Specialized High School Applicants Program 1 to 6` = sum(c_across(applicants1specialized:applicants6specialized), na.rm = TRUE),
    `Total True General Education Applicants Program 1 to 11` = sum(c_across(grade9geapplicants1:grade9geapplicants11), na.rm = TRUE),
    `Total True Applicants with Disabilities Program 1 to 11` = sum(c_across(grade9swdapplicants1:grade9swdapplicants11), na.rm = TRUE),
    `Total True Applicants` = sum(c_across(applicants1specialized:grade9swdapplicants11), na.rm = TRUE),
  ) %>%
  ungroup()

# Keep subset
directory_data_by_school_02 <- directory_data_by_school_02 %>%
  select(dbn, 
         `Total True Specialized High School Applicants Program 1 to 6`,
         `Total True General Education Applicants Program 1 to 11`,
         `Total True Applicants with Disabilities Program 1 to 11`,
         `Total True Applicants`)

## Join ------------------------------------------------------------------------
# 431

# Join 
# Directory data
data_02 <- applications_by_school_02 %>%
  left_join(directory_data_by_school_02,
            by = c("School DBN" = "dbn"))

# Remove specialized HS
data_02 <- data_02 %>%
  filter(!`School DBN` %in% specialized_HS)

# Add data that shows where Total True Applicants unequal to Grade 9 True Applicants
data_02 <- data_02 %>%
  mutate(
    `Total True Applicants unequal Grade 9 True Applicants` = if_else(`Total True Applicants` != `Grade 9 True Applicants`, 1, 0)
  )

# Save
write_xlsx(data_02, "DataGeneration/NYC/outputs/02_school_applicants.xlsx")

## Validation Check -------------------------------------------------------------

# Total true applicants != Grade 9 true applicants
unequal <- data_02 %>%
  filter(`Total True Applicants` != `Grade 9 True Applicants`)
# TRUE: 113

# Total true applicants < Grade 9 true applicants
lower <- data_02 %>%
  filter(`Total True Applicants` < `Grade 9 True Applicants`)
# TRUE: 10

# Total true applicants > Grade 9 true applicants
higher <- data_02 %>%
  filter(`Total True Applicants` > `Grade 9 True Applicants`)
# TRUE: 103
