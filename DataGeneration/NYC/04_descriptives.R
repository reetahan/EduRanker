########################### Descriptives Master Data ###########################
# Aim: Create descriptives to compare the four master data sets (by borough,
# residential district, zip code & home language)

# Follow: https://docs.google.com/presentation/d/17aHrpMHus9-3tf_bncJXiavLQyBsLuVKAkil1kVQ1i4/edit?slide=id.p#slide=id.p

# Packages 
library(readxl)
library(writexl)
library(ggplot2)
library(dplyr)
library(purrr)
library(tidyr)
library(stringr)
library(proxy)
library(vegan)
library(tibble)



# Load data --------------------------------------------------------------------

# Set path to data folder
# The folder should contain the downloaded data sets
data_path <- Sys.getenv("DATA_PATH_NYC")

# Load functions
source("DataGeneration/NYC/src/rbo.R")

# Load data sets

## Master Data Set: Boroughs
master_data_03_borough <- read_excel("DataGeneration/NYC/outputs/master_data_03_borough.xlsx")

## Master Data Set: Residential District
master_data_03_residential_district <- read_excel("DataGeneration/NYC/outputs/master_data_03_residential_district.xlsx")

## Master Data Set: Zip Code
master_data_03_zip_code <- read_excel("DataGeneration/NYC/outputs/master_data_03_zip_code.xlsx")

## Master Data Set: Home language
master_data_03_home_language <- read_excel("DataGeneration/NYC/outputs/master_data_03_home_language.xlsx")


# Descriptives -----------------------------------------------------------------

## Borough --------------------------------------------------------------------

# ! Remove "Unknown" for now because of tied ranks

master_data_03_borough <- master_data_03_borough %>%
  filter(`Borough` != "Unknown")

### Frequency Table ------------------------------------------------------------

# Question: 
# Which 15 schools appear most frequently in boroughs’ top-12 rankings, 
# and how highly are they ranked on average?

# Frequency Table

# 1) Create a table
top15_school_freq_avg_rank <- master_data_03_borough %>%
  filter(Rank <= 12) %>%
  distinct(`Borough`, `School Name`, Rank) %>%  # avoid double counting within district
  group_by(`School Name`) %>%
  summarise(
    `Frequency` = n_distinct(`Borough`),
    `Average Rank`    = mean(Rank, na.rm = TRUE)
  ) %>%
  arrange(desc(`Frequency`), `Average Rank`) %>%  # tie-break: better (lower) average rank first
  slice_head(n = 15)

### Similarity  ----------------------------------------------------------------

#### Question 1 ----------------------------------------------------------------
# Do boroughs rank the same schools within their top-12 in a similar order?

# Visual Representation

# 1) Keep only top-12 ranks per borough, clean strings
top12 <- master_data_03_borough %>%
  transmute(
    Borough = str_squish(as.character(Borough)),
    `School Name` = str_squish(as.character(`School Name`)),
    Rank = as.integer(Rank)
  ) %>%
  filter(!is.na(Borough), !is.na(`School Name`), !is.na(Rank)) %>%
  filter(Rank >= 1, Rank <= 12) %>%
  distinct(Borough, `School Name`, Rank)

# 2) Order schools: appear in many boroughs first, then by best rank
school_levels <- top12 %>%
  group_by(`School Name`) %>%
  summarise(
    n_boroughs = n_distinct(Borough),
    best_rank  = min(Rank),
    .groups = "drop"
  ) %>%
  arrange(desc(n_boroughs), best_rank, `School Name`) %>%
  pull(`School Name`)

# 3) Full grid (all boroughs x all schools) + join ranks
plot_data <- expand_grid(
  Borough = sort(unique(top12$Borough)),
  `School Name` = school_levels
) %>%
  left_join(top12, by = c("Borough", "School Name")) %>%
  mutate(
    Borough = factor(Borough, levels = sort(unique(top12$Borough))),
    `School Name` = factor(`School Name`, levels = rev(school_levels))  # top at top
  )

# 4) Plot: fill by Rank, NAs grey; fixed limits 1..12 for comparability
ggplot(plot_data, aes(x = Borough, y = `School Name`)) +
  geom_tile(aes(fill = Rank), color = "white", linewidth = 0.25) +
  geom_text(aes(label = ifelse(is.na(Rank), "", Rank)), size = 3) +
  scale_fill_gradientn(
    colours = c("#b2182b", "#f7f7f7", "#2166ac"),  # blue -> white -> red
    limits = c(1, 12),
    breaks = 1:12,
    na.value = "grey90",
    name = "Rank"
  ) +
  labs(x = "Borough", y = "School") +
  theme_minimal() +
  theme(
    panel.grid = element_blank(),
    axis.text.y = element_text(size = 7),
    axis.text.x = element_text(angle = 25, hjust = 1)
  )

# ggsave("DataGeneration/NYC/outputs/figures/descriptives_master_borough_rank_heatmap.png", width = 10, height = 6)

#### Question 2 ----------------------------------------------------------------
# Which boroughs select similar top-12 schools?

# Jaccard Similarity Index

# 1) Convert to a binary incidence matrix
incidence <- top12 %>%
  select(-Rank) %>%
  distinct(Borough, `School Name`) %>%   # VERY important
  mutate(value = 1) %>%
  pivot_wider(
    names_from  = `School Name`,
    values_from = value,
    values_fill = 0
  )
mat <- incidence %>%
  select(-Borough) %>%
  as.matrix()
rownames(mat) <- incidence$Borough


# 2) Compute Jaccard distance & similarity
jaccard_dist <- vegdist(mat, method = "jaccard", binary = TRUE)
jaccard_sim  <- 1 - as.matrix(jaccard_dist)


# 3) Create heatmap of Jaccard Similarity
jaccard_df <- as.data.frame(jaccard_sim) %>%
  rownames_to_column("Borough1") %>%
  pivot_longer(-Borough1, names_to = "Borough2", values_to = "Similarity")

ggplot(jaccard_df, aes(Borough1, Borough2, fill = Similarity)) +
  geom_tile(color = "white") +
  scale_fill_gradient(low = "white", high = "steelblue") +
  coord_equal() +
  labs(
    title = "Jaccard Similarity of Top-12 Selected Schools Between Boroughs",
    x = NULL,
    y = NULL,
    fill = "Jaccard"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# ggsave("DataGeneration/NYC/outputs/figures/descriptives_master_borough_similarity_schools_jaccard.png", width = 10, height = 6)

# 4) Create dendrogram based on Jaccard Similarity
jaccard_dist <- as.dist(1 - jaccard_sim)

hc <- hclust(jaccard_dist, method = "average")

plot(
  hc,
  main = "Borough Clustering Based on Jaccard Similarity",
  ylab = "1 − Jaccard similarity",
  xlab = ""
)


#### Question 3 ----------------------------------------------------------------
# Which boroughs rank their top-12 schools in a similar order?

# Rank Biased Overlap (RBO)
# Remark: Use RBO instead of Kendalls Tau since it is more robust for 
# partial overlaps!

# 1) Keep only top-12 ranks per borough, clean strings
boroughs <- sort(unique(top12$Borough))
rank_lists <- lapply(boroughs, function(b) {
  top12 %>%
    filter(Borough == b) %>%
    arrange(Rank) %>%
    pull(`School Name`)
})
names(rank_lists) <- boroughs

# 2) Create pairwise RBO similarity matrix
rbo_mat <- matrix(NA_real_, length(boroughs), length(boroughs),
                  dimnames = list(boroughs, boroughs))

for (i in seq_along(boroughs)) {
  for (j in seq_along(boroughs)) {
    rbo_mat[i, j] <- rbo_topk(rank_lists[[i]], rank_lists[[j]], p = 0.9, k = 12)
  }
}

# 5) Cluster + plot (distance = 1 - similarity)
rbo_dist <- as.dist(1 - rbo_mat)
hc <- hclust(rbo_dist, method = "average")

plot(
  hc,
  main = "Borough Clustering Based on Ranking Similarity (RBO)",
  ylab = "1 \u2212 RBO",
  xlab = ""
)

### Residential District -------------------------------------------------------

# ! Remove "Unknown" for now because of tied ranks

master_data_03_residential_district <- master_data_03_residential_district %>%
  filter(`Residential District` != "Unknown")

# Top 15 schools most frequently appearing in the top-12 rankings across residential districts, with average rank

top15_school_freq_avg_rank <- master_data_03_residential_district %>%
  filter(Rank <= 12) %>%
  distinct(`Residential District`, `School Name`, Rank) %>%  # avoid double counting within district
  group_by(`School Name`) %>%
  summarise(
    `Frequency` = n_distinct(`Residential District`),
    `Average Rank`    = mean(Rank, na.rm = TRUE)
  ) %>%
  arrange(desc(`Frequency`), `Average Rank`) %>%  # tie-break: better (lower) average rank first
  slice_head(n = 15)


# Do residential districts rank the same schools within their top-12 in a similar order?

# 1) Keep only top-12 ranks per borough, clean strings
top12 <- master_data_03_residential_district %>%
  transmute(
    `Residential District` = str_squish(as.character(`Residential District`)),
    `School Name` = str_squish(as.character(`School Name`)),
    Rank = as.integer(Rank)
  ) %>%
  filter(!is.na(`Residential District`), !is.na(`School Name`), !is.na(Rank)) %>%
  filter(Rank >= 1, Rank <= 12) %>%
  distinct(`Residential District`, `School Name`, Rank)

# 2) Order schools: appear in many res districts first, then by best rank
school_levels <- top12 %>%
  group_by(`School Name`) %>%
  summarise(
    n_res = n_distinct(`Residential District`),
    best_rank  = min(Rank),
    .groups = "drop"
  ) %>%
  arrange(desc(n_res), best_rank, `School Name`) %>%
  pull(`School Name`)

# 3) Full grid (all boroughs x all schools) + join ranks
plot_data <- expand_grid(
  `Residential District` = sort(unique(top12$`Residential District`)),
  `School Name` = school_levels
) %>%
  left_join(top12, by = c("Residential District", "School Name")) %>%
  mutate(
    `Residential District` = factor(`Residential District`, levels = sort(unique(top12$`Residential District`))),
    `School Name` = factor(`School Name`, levels = rev(school_levels))  # top at top
  )

# 4) Plot: fill by Rank, NAs grey; fixed limits 1..12 for comparability
ggplot(plot_data, aes(x = `Residential District`, y = `School Name`)) +
  geom_tile(aes(fill = Rank), color = "white", linewidth = 0.25) +
  geom_text(aes(label = ifelse(is.na(Rank), "", Rank)), size = 3) +
  scale_fill_gradientn(
    colours = c("#b2182b", "#f7f7f7", "#2166ac"),  # blue -> white -> red
    limits = c(1, 12),
    breaks = 1:12,
    na.value = "grey90",
    name = "Rank"
  ) +
  labs(x = "Residential District", y = "School") +
  theme_minimal() +
  theme(
    panel.grid = element_blank(),
    axis.text.y = element_text(size = 7),
    axis.text.x = element_text(angle = 25, hjust = 1)
  )

# 5) Save plot
#ggsave("DataGeneration/NYC/outputs/figures/descriptives_master_res_dis_rank_heatmap.png", width = 10, height = 6)





