required_packages <- c(
  "data.table",
  "dplyr",
  "lubridate",
  "clustMixType",
  "dbscan"
)

missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0) {
  stop(
    paste(
      "Install the missing packages before running this script:",
      paste(missing_packages, collapse = ", ")
    )
  )
}

library(data.table)
library(dplyr)
library(lubridate)
library(clustMixType)
library(dbscan)

set.seed(597)

collapse_weather <- function(x, top_n = 8) {
  x <- ifelse(is.na(x) | x == "", "Missing", x)
  top_levels <- names(sort(table(x), decreasing = TRUE))[seq_len(min(top_n, length(unique(x))))]
  ifelse(x %in% top_levels, x, "Other")
}

hour_band_from_hour <- function(hour_value) {
  dplyr::case_when(
    is.na(hour_value) ~ "Unknown",
    hour_value >= 6 & hour_value <= 9 ~ "AM Rush",
    hour_value >= 10 & hour_value <= 13 ~ "Midday",
    hour_value >= 14 & hour_value <= 18 ~ "PM Rush",
    hour_value >= 19 & hour_value <= 23 ~ "Evening",
    TRUE ~ "Overnight"
  )
}

season_from_month <- function(month_value) {
  dplyr::case_when(
    is.na(month_value) ~ "Unknown",
    month_value %in% c(12, 1, 2) ~ "Winter",
    month_value %in% c(3, 4, 5) ~ "Spring",
    month_value %in% c(6, 7, 8) ~ "Summer",
    TRUE ~ "Fall"
  )
}

project_coords_km <- function(lat, lng) {
  mean_lat_rad <- mean(lat, na.rm = TRUE) * pi / 180
  lat_km <- lat * 111.32
  lng_km <- lng * 111.32 * cos(mean_lat_rad)
  data.frame(lat_km = lat_km, lng_km = lng_km)
}

raw_data <- fread("../li_accidents_clean.csv", na.strings = c("", "NA"))

li_data <- raw_data %>%
  mutate(
    Start_Time = ymd_hms(Start_Time, quiet = TRUE),
    hour = hour(Start_Time),
    month = month(Start_Time),
    weekday = wday(Start_Time, label = TRUE, abbr = FALSE),
    weekend = factor(if_else(weekday %in% c("Saturday", "Sunday"), "Weekend", "Weekday")),
    hour_band = factor(hour_band_from_hour(hour)),
    season = factor(season_from_month(month)),
    rush_hour = factor(if_else(hour_band %in% c("AM Rush", "PM Rush"), "Rush", "Non-Rush")),
    weather_group = factor(collapse_weather(Weather_Condition)),
    Sunrise_Sunset = factor(if_else(is.na(Sunrise_Sunset), "Missing", Sunrise_Sunset)),
    Crossing = factor(Crossing),
    Junction = factor(Junction),
    Traffic_Signal = factor(Traffic_Signal),
    Stop = factor(Stop),
    Railway = factor(Railway),
    Amenity = factor(Amenity),
    Station = factor(Station),
    County = factor(County),
    Severity = factor(Severity),
    `Temperature(F)` = as.numeric(`Temperature(F)`),
    `Humidity(%)` = as.numeric(`Humidity(%)`),
    `Pressure(in)` = as.numeric(`Pressure(in)`),
    `Visibility(mi)` = as.numeric(`Visibility(mi)`),
    `Wind_Speed(mph)` = as.numeric(`Wind_Speed(mph)`),
    Start_Lat = as.numeric(Start_Lat),
    Start_Lng = as.numeric(Start_Lng)
  )

# Scenario clustering intentionally excludes Severity and raw location.
scenario_data <- li_data %>%
  transmute(
    hour_band,
    weekday = factor(weekday),
    weekend,
    season,
    rush_hour,
    Sunrise_Sunset,
    weather_group,
    `Temperature(F)`,
    `Humidity(%)`,
    `Pressure(in)`,
    `Visibility(mi)`,
    `Wind_Speed(mph)`,
    Crossing,
    Junction,
    Traffic_Signal,
    Stop,
    Railway,
    Amenity,
    Station
  )

message("Scenario clustering validation for k = 3:8")

# Review this object and choose a best k using the reported validation output.
validation_result <- validation_kproto(
  object = NULL,
  x = scenario_data,
  k = 3:8,
  type = "gower",
  nstart = 5,
  na.rm = "imp.internal",
  verbose = TRUE
)

print(validation_result)

# Update this value after reviewing validation output.
best_k <- 4L

message("Fitting final k-prototypes model with best_k = ", best_k)

scenario_fit <- kproto(
  x = scenario_data,
  k = best_k,
  type = "gower",
  nstart = 10,
  na.rm = "imp.internal",
  keep.data = TRUE,
  verbose = TRUE
)

print(summary(scenario_fit))

message("Running bootstrap stability analysis")

stability_result <- stability_kproto(
  object = scenario_fit,
  method = c("jaccard", "rand"),
  B = 25,
  verbose = TRUE
)

print(stability_result)

li_data$scenario_cluster <- factor(scenario_fit$cluster)

scenario_profile <- li_data %>%
  count(scenario_cluster, County, Severity, sort = TRUE)

fwrite(scenario_profile, "Li_ScenarioCluster_Profile.csv")

coords <- project_coords_km(li_data$Start_Lat, li_data$Start_Lng)

message("Inspect the kNN distance plot and adjust eps if needed")
kNNdistplot(coords, k = 25)
abline(h = 0.35, col = "red", lty = 2)

# Replace eps after visually inspecting the kNN distance elbow.
hotspot_fit <- dbscan(coords, eps = 0.35, minPts = 25)

li_data$hotspot_cluster <- hotspot_fit$cluster

hotspot_summary <- li_data %>%
  count(hotspot_cluster, County, sort = TRUE)

fwrite(hotspot_summary, "Li_Hotspot_Profile.csv")
fwrite(as.data.table(li_data), "Li_Accidents_Clustered.csv")

message("Clustering pipeline finished.")
