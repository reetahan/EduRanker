########################### Fall 2024: SAE Algorithm #########################
# Aim: Create data set as for NYC school-student matching
# (1): Individual-level preference list with priorities + assigned school
# (2): By region matching outcomes
# (3): School capacity

# Remark: Focus only on HS students (Grade 9)

# Data Source:
# https://datosabiertos.mineduc.cl/sistema-de-admision-escolar-sae/
# Relevant data sets are:
# B1_Postulantes_etapa_regular_2024_Admisión_2025_PU
# C1_Postulaciones_etapa_regular_2024_Admisión_2025_PUB
# D1_Resultados_etapa_regular_2024_Admisión_2025_PUB
# A1_Oferta_Establecimientos_etapa_regular_2024_Admisión_2025

# Packages
library(ggplot2)
library(readr)
library(sf)
library(dplyr)
library(writexl)


# Load data --------------------------------------------------------------------

# Set path
# The folder should contain the downloaded data sets
data_path <- Sys.getenv("DATA_PATH_CHILE")

# Load functions

# Load data sets

# 1) Information about applicants
# Rows: 473.482
# Grade 9: 124.647 (For later, without NAs in Region: 124.492)
postulantes <- read_csv2(paste0(data_path,"/B1_Postulantes_etapa_regular_2024_Admisión_2025_PUBL.csv"))
postulantes <- postulantes %>%
  filter(cod_nivel == 9) 

# 2) Preference list of applicants
# Rows: 1.638.432
# Grade 9: 482.204 (For later, without NAs in Region: 481.678)
postulaciones <- read_csv2(paste0(data_path,"/C1_Postulaciones_etapa_regular_2024_Admisión_2025_PUBL.csv"))
postulaciones <- postulaciones %>%
  filter(cod_nivel == 9) 
  
# 3) Application results
# Rows: 473.482
# Grade 9: 124.647
resultados <- read_csv2(paste0(data_path,"/D1_Resultados_etapa_regular_2024_Admisión_2025_PUBL.csv"))
resultados <- resultados %>%
  filter(cod_nivel == 9)

# 4) Geo regional & provinces data Chile
# Source: https://www.bcn.cl/siit/mapas_vectoriales/index_html
shp <- st_read(paste0(data_path,"/geo/Regiones/Regional.shp"), quiet = TRUE) %>%
  select(objectid, Region)

shp_province <- st_read(paste0(data_path,"/geo/Provincias/Provincias.shp"), quiet = TRUE) %>%
  select(objectid, Provincia)


# 5) School capacity
# Grade 9: 79.162
capacidad <- read_csv2(paste0(data_path,"/A1_Oferta_Establecimientos_etapa_regular_2024_Admisión_2025.csv"))
capacidad <- capacidad %>%
  filter(cod_nivel == 9)

# Create data sets -------------------------------------------------------------

## (1): Individual-level preference list  --------------------------------------

### STEP 1: GEO DATA -----------------------------------------------------------

####  Regions ------------------------------------------------------------------

# Make polygons valid and keep only needed columns
shp <- shp %>%
  st_make_valid() %>%
  select(objectid, Region)

# keep all rows from postulantes
# CRS in shp is not  4362, but 5360
postulantes_sf <- postulantes %>%
  st_as_sf(
    coords = c("lon_con_error", "lat_con_error"),
    crs = 4326,
    remove = FALSE,
    na.fail = FALSE
  ) %>%
  st_transform(st_crs(shp))

# Find intersection
idx <- st_intersects(postulantes_sf, shp)
region_idx <- sapply(idx, function(x) if (length(x) == 0) NA_integer_ else x[1])

postulantes_sf$objectid <- shp$objectid[region_idx]
postulantes_sf$Region   <- shp$Region[region_idx]

# Back to df format
# Remark: For now, 155 NAs
postulantes_region_df <- postulantes_sf %>%
  st_drop_geometry()

# Validate
# Compare to official regional stats reported for Kidnergarten applications in:
# https://accioneducar.cl/wp-content/uploads/2025/05/Analisis-resultados-SAE-2025-MVV-VF_250520_150037.pdf
#postulantes_region_df_kinder <- postulantes_region_df %>%
  #filter(cod_nivel == 0)
#table(postulantes_region_df_kinder$Region)


####  Provinces ----------------------------------------------------------------

# Make polygons valid and keep only needed columns
shp_province <- shp_province %>%
  st_make_valid() %>%
  select(objectid, Provincia)

# keep all rows from postulantes
# CRS in shp is not  4362, but 5360
postulantes_province_region_df <- postulantes_region_df %>%
  st_as_sf(
    coords = c("lon_con_error", "lat_con_error"),
    crs = 4326,
    remove = FALSE,
    na.fail = FALSE
  ) %>%
  st_transform(st_crs(shp_province))

# Find intersection
idx <- st_intersects(postulantes_province_region_df, shp_province)
province_idx <- sapply(idx, function(x) if (length(x) == 0) NA_integer_ else x[1])

postulantes_province_region_df$objectid <- shp_province$objectid[province_idx]
postulantes_province_region_df$Provincia   <- shp_province$Provincia[province_idx]

# Back to df format
# Remark: For now, 155 NAs
postulantes_province_region_df <- postulantes_province_region_df %>%
  st_drop_geometry()

# Validate

# Check regions
# ggplot(postulantes_province_region_df,
#        aes(x = lon_con_error, y = lat_con_error, color = Region)) +
#   geom_point(alpha = 0.4, size = 0.5) +
#   coord_fixed() +
#   theme_minimal()

# # Check that provinces are correctly within the regions
# postulantes_province_region_df %>%
#   distinct(Provincia, Region) %>%
#   count(Provincia) %>%
#   filter(n > 1)
# 
# postulantes_province_region_df %>%
#   distinct(Provincia, Region) %>%
#   count(Provincia, name = "n_regions") %>%
#   arrange(desc(n_regions))
# 
# postulantes_province_region_df %>%
#   distinct(Region, Provincia) %>%
#   count(Region, name = "n_provinces") %>%
#   ggplot(aes(x = reorder(Region, n_provinces), y = n_provinces)) +
#   geom_col() +
#   coord_flip() +
#   labs(
#     x = "Region",
#     y = "Number of Provinces",
#     title = "Number of Provinces per Region (from your data)"
#   ) +
#   theme_minimal()

### STEP 2: Preference DATA ---------------------------------------------------

# Subset both data sets
# Data set with applicant info &
# data set with preference info

applicants_HS <- postulantes_province_region_df %>%
  select(- cod_nivel)

preferences_applicants_HS <- postulaciones %>%
  select(- cod_nivel, -orden_pie, -orden_alta_exigencia_transicion)

# Join
# Join demographic, priority and geo data to the preference data 
preferences_applicants_HS <- preferences_applicants_HS %>%
  left_join(applicants_HS, by = "mrun")

# Remove observations where Region is NA (155 observations)
# Rows: 481.678
preferences_applicants_HS <- preferences_applicants_HS %>%
  filter(!is.na(Region))

#rm(applicants_HS, idx, postulantes_region_df, postulantes_sf, shp)

### STEP 3: Matching DATA ------------------------------------------------------

# Add final matching outcome to the preference data set

# Create empty column "matched"
# therefore join with resultados
# "matched" takes the value 1 for the match with resultados by mrun, cod_curso_admitido & rbd_admitido

preferences_applicants_HS <- preferences_applicants_HS %>%
  left_join(
    resultados %>%
      select(mrun, rbd_admitido, cod_curso_admitido) %>%
      distinct() %>%
      mutate(matched = 1),
    by = c("mrun", "rbd" = "rbd_admitido", "cod_curso" = "cod_curso_admitido")
  ) %>%
  mutate(matched = ifelse(is.na(matched), 0, 1))

## (2): Matching outcome  ------------------------------------------------------

# Create final matching outcome table
# Columns:
# Region
# Total number of students
# % matched to 1
# % matched to 2
# ... and so on
# % not matched (where rbd_admitido == 0)


### Regions --------------------------------------------------------------------

# Remarks:
# 14.504 are in total unmatched (in resultados it is 14.515 given that there
# are still students with missing region)

student_outcomes <- preferences_applicants_HS %>%
  group_by(Region, mrun) %>%
  summarise(
    matched_pref = if (any(matched == 1)) {
      min(preferencia_postulante[matched == 1], na.rm = TRUE)
    } else {
      NA_real_
    },
    .groups = "drop"
  )

final_matching_outcome <- student_outcomes %>%
  group_by(Region) %>%
  summarise(
    n_students = n(),
    !!!setNames(
      lapply(1:21, function(k) {
        rlang::expr(mean(matched_pref == !!k, na.rm = TRUE) * 100)
      }),
      paste0("pct_top", 1:21)
    ),
    pct_unmatched = mean(is.na(matched_pref)) * 100,
    .groups = "drop"
  )


### Provinces -----------------------------------------------------------------

# Remarks:
# 14.504 are in total unmatched (in resultados it is 14.515 given that there
# are still stundents with missing region)

student_outcomes_province <- preferences_applicants_HS %>%
  group_by(Provincia, mrun) %>%
  summarise(
    matched_pref = if (any(matched == 1)) {
      min(preferencia_postulante[matched == 1], na.rm = TRUE)
    } else {
      NA_real_
    },
    .groups = "drop"
  )

final_matching_outcome_province <- student_outcomes_province %>%
  group_by(Provincia) %>%
  summarise(
    n_students = n(),
    !!!setNames(
      lapply(1:21, function(k) {
        rlang::expr(mean(matched_pref == !!k, na.rm = TRUE) * 100)
      }),
      paste0("pct_top", 1:21)
    ),
    pct_unmatched = mean(is.na(matched_pref)) * 100,
    .groups = "drop"
  )



## (3): School Capacity  -------------------------------------------------------

# Data 1:
# Keep relevant rows & columns
school_capacity <- capacidad %>%
  select(rbd, cod_curso, lat,lon,cupos_totales,vacantes,vacantes_pie,
         vacantes_prioritarios, vacantes_alta_exigencia_t,
         vacantes_alta_exigencia_r, vacantes_regular)


# Data 2:
# Use resultados and count number of students by region admitted to each school (rbd_admitido) and course (cod_curso_admitido)
# in the data preferences_applicants_HS

school_capacity_by_region <- preferences_applicants_HS %>%
  filter(matched == 1) %>%
  group_by(Region, rbd, cod_curso) %>%
  summarise(n_admitted = n(), .groups = "drop")

# to validate count number of students by reggion
#school_capacity_by_region %>%
 # group_by(Region) %>%
  #summarise(n_admitted = sum(n_admitted)) 

# Save data  -----------------------------------------------------------------

# Rename data sets
preferences_applicants_HS <- preferences_applicants_HS %>%
  rename(
    program_code = cod_curso,
    preference_number = preferencia_postulante,
    priority_already_registered = prioridad_matriculado,
    priority_sibling = prioridad_hermano,
    priority_parent_civil_servant = prioridad_hijo_funcionario,
    priority_ex_student = prioridad_exalumno,
    lottery = loteria_original,
    integration_program_status_existing = es_pie,
    female = es_mujer,
    priority_student = prioritario,
    high_performance_student = alto_rendimiento,
    matched_first_round = matched,
    province = Provincia,
    quality_georef = calidad_georef,
    lon = lon_con_error,
    lat = lat_con_error
  ) %>%
  select(
    mrun, Region, province, lon, lat, quality_georef, rbd, program_code, lottery, preference_number,
    matched_first_round,
    female, priority_student, high_performance_student,
    priority_already_registered, priority_sibling,
    priority_parent_civil_servant, priority_ex_student,
    integration_program_status_existing
  )
       
school_capacity_by_region <- school_capacity_by_region %>%
  rename(
    program_code = cod_curso
  ) %>%
  select(Region, rbd, program_code, n_admitted)


school_capacity <- school_capacity %>%
  rename(
    program_code = cod_curso,
    total_capacity = cupos_totales,
    total_admission_seats= vacantes,
    integration_student_seats = vacantes_pie,
    priority_student_seats= vacantes_prioritarios,
    high_selectivity_seats_transitional= vacantes_alta_exigencia_t,
    high_selectivity_seats_ranking= vacantes_alta_exigencia_r,
    regular_seats= vacantes_regular
  ) %>%
  select(rbd, program_code, lat, lon,total_capacity, total_admission_seats,
         integration_student_seats, priority_student_seats,
         high_selectivity_seats_transitional, high_selectivity_seats_ranking,
         regular_seats )

# Save as csv

write_xlsx(preferences_applicants_HS, "DataGeneration/Chile/outputs/individual_level_preferences_and_result.xlsx")
write_xlsx(preferences_applicants_HS, "DataGeneration/Chile/outputs/individual_level_preferences_and_result_province.xlsx")
write_xlsx(final_matching_outcome, "DataGeneration/Chile/outputs/matching_outcome_by_region.xlsx")
write_xlsx(final_matching_outcome_province, "DataGeneration/Chile/outputs/matching_outcome_by_province.xlsx")
write_xlsx(school_capacity_by_region, "DataGeneration/Chile/outputs/school_capacity_by_region.xlsx")
write_xlsx(school_capacity, "DataGeneration/Chile/outputs/school_capacity.xlsx")



