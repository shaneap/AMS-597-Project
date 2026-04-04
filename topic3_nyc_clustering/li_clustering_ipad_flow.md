# Long Island Clustering Workflow

Use this like a flowchart on your iPad.

## One-line idea

`Clean Long Island accidents` -> `build features` -> `run 2 clustering tracks` -> `join results back` -> `interpret` -> `export figures and tables`

## Full flow

### 1) Start with the cleaned file

`li_accidents_clean.csv`

- This is the shared Long Island dataset.
- Every later step depends on this file.

### 2) Audit the data

`Dataset checks`

- Check row count, county count, severity count, and missingness.
- Check for odd values like zero-distance rows or sparse road flags.
- Goal: make sure the data is trustworthy before modeling.

### 3) Build analysis features

`Raw columns` -> `usable clustering features`

- Time becomes: `hour_band`, `weekend`, `season`
- Weather becomes: `weather_group`
- Wind becomes: `wind_dir_norm`
- Duration becomes: `log_Duration`

Why:

- Clustering works better when the inputs are cleaner and easier to compare.

### 4) Split into two tracks

`Feature set` -> `Scenario track` + `Spatial track`

#### A. Scenario track

`df_scenario`

Uses:

- hour band
- weather group
- weekend
- road-context flags
- visibility
- temperature
- wind speed
- log duration

Model:

- `k-prototypes`

Why:

- This model handles mixed data.
- Some variables are numeric.
- Some variables are categorical.

Question answered:

- “What kinds of accident situations exist?”

#### B. Spatial track

`df_spatial`

Uses:

- projected Long Island coordinates only

Model:

- `DBSCAN`

Why:

- It is good for hotspot detection.
- It does not force every row into a cluster.
- It can leave isolated points as noise.

Question answered:

- “Where do accidents physically cluster on Long Island?”

### 5) Validate the scenario track

`candidate k values` -> `consensus choice of k`

- Run several candidate `k-prototypes` solutions.
- Check silhouette fit.
- Check bootstrap stability.
- On a 16 GB machine, do this with repeated samples and combine the results.

Why:

- We want a scenario solution that is not just mathematically possible, but also reasonably repeatable.

### 6) Select the final scenario clusters

`validation + stability` -> `selected scenario model`

- Pick the most defensible value of `k`
- Save the final scenario cluster labels

Current meaning:

- This gives the broad accident archetypes.

### 7) Select the final spatial hotspots

`DBSCAN search` -> `final hotspot map`

- Try several `eps` and `minPts` combinations.
- Keep solutions that fit the project targets.
- Reject solutions where one hotspot becomes unrealistically huge.

Current meaning:

- This gives the Long Island hotspot corridors.

### 8) Join both results back to each accident row

`Scenario cluster` + `Spatial cluster` -> `df_clustered`

- Every accident gets:
  - a scenario cluster label
  - a spatial cluster label

Why:

- This is what lets us compare crash type and crash location together.

### 9) Interpret the clusters

`df_clustered` -> `meaningful summaries`

- Severity by scenario cluster
- County mix by scenario cluster
- Weather/time profile by scenario cluster
- Dominant scenario inside each hotspot

This is where the project turns into findings.

### 10) Export presentation material

`Results` -> `figures + tables + interactive HTML`

- hotspot map
- density heatmap
- scenario risk bubble chart
- cluster fingerprint heatmap
- spatial-scenario heatmap
- labeled clustered CSV

### 11) Final sanity audit

`Outputs` -> `sanity checks`

Check:

- scenario names are unique
- no missing labels
- spatial solution is balanced
- stability is honest enough to report correctly

## Fast explanation of the models

### `k-prototypes`

Use this when the data mixes:

- categories like `weather_group`
- numbers like `visibility`

It groups accidents into broad situation-types.

### `DBSCAN`

Use this when the data is just location.

It finds dense hotspots and leaves isolated points as noise.

## Fast interpretation of the final project

- `Scenario clustering` = accident situations
- `Spatial clustering` = accident locations
- `Joined results` = which situations happen in which hotspots

## Best way to explain it aloud

“First, the workflow checks and prepares the Long Island accident data. Then it runs one clustering model to group accidents by situation and another clustering model to group them by location. After that, it joins both sets of labels back to the data, summarizes severity and county patterns, and exports the final maps, heatmaps, and report-ready outputs.”
