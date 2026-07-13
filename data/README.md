# Data directory

The analysis downloads the source data at runtime and creates `complaints_raw.csv` and `complaints_clean.csv` locally.

These intermediate files are excluded from version control because they are reproducible, comparatively large, and contain real public complaint narratives.

Run the complete notebook to regenerate them:

`notebooks/consumer_complaints_kmeans.ipynb`
