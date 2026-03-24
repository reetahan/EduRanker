######################## Fall 2024: School Capacity  ############################
# AIM:
# Create data set
# (1): Capacity by school: Offers, Seats & Student enrollment

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

# DATA 01 ----------------------------------------------------------------------
# Capacity by school data set

## Prepare ---------------------------------------------------------------------

## 1.1)

# Keep relevant columns
applications_by_school_01 <- applications_by_school %>%
  select(`School District`, `School DBN`, `School Name`,
         `Category`,`Grade 9 Seats Available` , `Grade 9 Offers`)


# Convert s to -1 and s^ to -2 
applications_by_school_01 <- applications_by_school_01 %>%
  mutate(
    `Grade 9 Seats Available`   = convert_col(`Grade 9 Seats Available`),
    `Grade 9 Offers`            = convert_col(`Grade 9 Offers`)
  )

# Keep category "All Students"
# Rows: 431
applications_by_school_01 <- applications_by_school_01 %>%
  filter(Category == "All Students") %>%
  filter(!is.na(`Grade 9 Offers`) & !is.na(`Grade 9 Seats Available`))


## 1.2)

# Keep relevant columns
enrollments_by_school_01 <- enrollments_by_school %>%
  select( `School DBN`, 
          `Category`, `Grade 9 Students`)

# Convert s to -1 and s^ to -2 
enrollments_by_school_01 <- enrollments_by_school_01 %>%
  mutate(
    `Grade 9 Students`   = convert_col(`Grade 9 Students`),
  )

# Keep category "All Students"
# Rows: 474
enrollments_by_school_01 <- enrollments_by_school_01 %>%
  filter(Category == "All Students") %>%
  filter(!is.na(`Grade 9 Students`)) 

# Check duplicates
duplicates <- enrollments_by_school_01  %>%
  filter(duplicated(across(c(`School DBN`, Category))) |
           duplicated(across(c(`School DBN`, Category)), fromLast = TRUE))

# Remove duplicates
# Keep highest number (tbd)
# Obs: 466
enrollments_by_school_01 <- enrollments_by_school_01 %>%
  group_by(`School DBN`, Category) %>%
  slice_max(`Grade 9 Students`, n = 1, with_ties = FALSE) %>%
  ungroup()

# 2.1)

# Keep relevant columns
directory_data_by_school_01 <- directory_data_by_school %>%
  select( dbn, seats1specialized, seats2specialized,
          seats3specialized, seats4specialized, 
          seats5specialized, seats6specialized,
          seats9ge1, seats9ge2,
          seats9ge3, seats9ge4,
          seats9ge5, seats9ge6,
          seats9ge7, seats9ge8,
          seats9ge9, seats9ge10,
          seats9ge11,
          seats9swd1, seats9swd2,
          seats9swd3, seats9swd4,
          seats9swd5, seats9swd6,
          seats9swd7, seats9swd8,
          seats9swd9, seats9swd10,
          seats9swd11)

# Remove NAs
# Rows: 444
directory_data_by_school_01 <- directory_data_by_school_01 %>%
  filter(!if_all(-dbn, ~ is.na(.x)))

# Convert to numeric
directory_data_by_school_01 <- directory_data_by_school_01 %>%
  mutate(across(-dbn, ~ suppressWarnings(as.numeric(.))))

# Create column
# Create a column that sums the number of seats of
# at Specialized HS, General Education HS,
# and Students with Disabilities HS
directory_data_by_school_01 <- directory_data_by_school_01 %>%
  rowwise() %>%
  mutate(
    `Total Seats at Specialized High School Program 1 to 6` = sum(c_across(seats1specialized:seats6specialized), na.rm = TRUE),
    `Total Seats for General Education Students Program 1 to 11` = sum(c_across(seats9ge1:seats9ge11), na.rm = TRUE),
    `Total Seats for Students with Disabilities Program 1 to 11` = sum(c_across(seats9swd1:seats9swd11), na.rm = TRUE),
    `Total Seats` = sum(c_across(seats1specialized:seats9swd11), na.rm = TRUE),
  ) %>%
  ungroup()

# Keep subset
directory_data_by_school_01_subset <- directory_data_by_school_01 %>%
  select(dbn, 
         `Total Seats at Specialized High School Program 1 to 6`,
         `Total Seats for General Education Students Program 1 to 11`,
         `Total Seats for Students with Disabilities Program 1 to 11`,
         `Total Seats`)

# 2.1.2)

# Keep relevant columns
directory_data_by_school_01_01 <- directory_data_by_school %>%
  select( dbn,   "priority1_prog1",  "priority2_prog1",
          "priority3_prog1",  "priority1_prog2",
          "priority2_prog2",  "priority3_prog2",
          "priority1_prog3",  "priority2_prog3",
          "priority3_prog3",  "priority1_prog4",
          "priority2_prog4",  "priority3_prog4",
          "priority1_prog5",  "priority2_prog5",
          "priority3_prog5",  "priority1_prog6",
          "priority2_prog6",  "priority3_prog6",
          "priority1_prog7",  "priority2_prog7",
          "priority1_prog8",  "priority2_prog8",
          "priority1_prog9",  "priority2_prog9",
          "priority1_prog10", "priority2_prog10",
          "priority1_prog11", "priority2_prog11",
          "diadetails")


# Add column "Priority"
# with 1 if any of the above is not NA
directory_data_by_school_01_01 <- directory_data_by_school_01_01 %>%
  mutate(
    Priority = if_any(c(starts_with("priority")), ~ !is.na(.))
  )


## Join ------------------------------------------------------------------------
# Rows: 431

# Join
# Enrollments
data_01 <- applications_by_school_01 %>%
  left_join(enrollments_by_school_01,
            by = c("School DBN" = "School DBN")) 

# Clean
# Rows: 431
data_01 <- data_01 %>%
  select(`School DBN`, `School District`, `School Name`,
         `Grade 9 Students`,
         `Grade 9 Seats Available`, `Grade 9 Offers`)

# Join
# Seats summed up by programs
data_01 <- data_01 %>%
  left_join(directory_data_by_school_01_subset,
            by = c("School DBN" = "dbn"))


# Add column with 1 if Offers > Seats Available
data_01 <- data_01 %>%
  mutate(
    `Offers > Grade 9 Seats Available` = if_else(`Grade 9 Offers` > `Grade 9 Seats Available`, 1, 0)
  )

# Add column with 1 # Offers > Total Seats (Sum of all seats by programs)
data_01 <- data_01 %>%
  mutate(
    `Offers > Total Seats` = if_else(`Grade 9 Offers` > `Total Seats`, 1, 0)
  )


# Rename 
data_01 <- data_01 %>%
  rename(
    `Grade 9 Total Seats Available by Program` = `Total Seats`
  ) %>%
  rename(
    `Offers > Grade 9 Total Seats Available by Program` = `Offers > Total Seats`)

# Remove Specialized HS
specialized_HS <- c("13K430","02M475","05M692", "10X445", "14K449", "28Q687", "10X696", "31R605", "03M485")

data_01 <- data_01 %>%
  filter(!`School DBN` %in% specialized_HS)


# Save
write_xlsx(data_01, "DataGeneration/NYC/outputs/01_school_capacity.xlsx")

## Validation Check ------------------------------------------------------------

# Offers > Seats Available
higher_01 <- data_01 %>%
  filter(`Grade 9 Offers` > `Grade 9 Seats Available`)
# TRUE: 194

# Offers > Total Seats (Sum of all seats by programs)
higher_02 <- data_01 %>%
  filter(`Grade 9 Offers` > `Total Seats`)
# TRUE: 189

# How many are in both?
sum(higher_01$"School DBN" %in% higher_02$"School DBN")
# TRUE: 187



