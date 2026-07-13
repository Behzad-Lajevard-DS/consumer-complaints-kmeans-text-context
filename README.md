# When Text Context Limits Clustering

## An exploratory NLP case study using TF–IDF and K-means on consumer financial complaints

**Author:** Behzad Lajevard  
**Project type:** Unsupervised Natural Language Processing  
**Dataset:** CFPB consumer complaint narratives from Hugging Face  
**Primary notebook:** [`notebooks/consumer_complaints_kmeans.ipynb`](notebooks/consumer_complaints_kmeans.ipynb)

## What this project is—and is not

This repository is **not** intended to prove that K-means performs successfully on consumer complaint narratives, and it is not presented as a production-ready topic classification system.

Its central purpose is methodological: to demonstrate that **text context, corpus structure, and problem formulation must be examined before a clustering model is selected**. A technically careful text-processing pipeline can remove noise, standardize narratives, and construct valid numerical features. It cannot create naturally separable clusters when the underlying texts are multi-topic, formulaic, company-specific, or semantically overlapping.

In this study, the low silhouette scores and the overlapping two-dimensional representation are not hidden as failures. They are the main empirical lesson. The pipeline shows that even precise preprocessing cannot compensate for a corpus that does not support hard thematic separation. Model construction should therefore follow a defensible research question and a diagnosis of the text—not blind algorithm selection.

## Project summary

The notebook streams the first 50,000 records from the public CFPB complaint dataset, detects the narrative and metadata columns, cleans the text, constructs TF–IDF unigram and bigram features, evaluates K-means solutions from K=2 to K=10, interprets the selected clusters, and exports all tables and figures.

| Item | Result |
|---|---:|
| Streamed source records | 50,000 |
| Usable complaint narratives | 1,984 |
| TF–IDF features | 7,374 |
| Candidate K values | 2–10 |
| Selected K within tested range | 10 |
| Best tested silhouette score | 0.0245 |
| Variance explained by two SVD components | 2.28% |

The selected value of K was the **upper boundary** of the tested range, and the silhouette score remained very low. K=10 should therefore be interpreted as the best solution *among the tested candidates*, not as proof of a globally optimal or naturally separated cluster structure.

## Main methodological lesson

> High-quality preprocessing improves the representation of text, but it does not guarantee that the corpus is clusterable. The analytical objective, unit of analysis, and modelling strategy must be grounded in the context of the text before a model is built.

This matters because individual complaints frequently contain several connected issues. A single narrative may mention an unauthorized transaction, a failed company investigation, account access, and incorrect credit reporting. K-means is a hard-clustering algorithm, so it must force that document into one cluster even when the document legitimately belongs to several themes.

## Pipeline

1. Stream a reproducible subset from Hugging Face.
2. Detect the narrative, product, issue, and company columns.
3. Remove missing, empty, very short, and exact-duplicate narratives.
4. Apply light text preprocessing.
5. Construct normalized TF–IDF unigram and bigram features.
6. Evaluate K using inertia and sampled silhouette scores.
7. Fit the final K-means model.
8. Inspect cluster sizes and centroid terms.
9. Retrieve representative complaints nearest to each centroid.
10. Compare clusters with product and issue metadata.
11. Create a two-dimensional Truncated SVD projection.
12. Export reproducible tables, figures, clustered data, and metadata.

## Key figures

### Narrative length distribution

![Narrative length distribution](outputs/figures/narrative_length_distribution.png)

### Elbow curve

![Elbow plot](outputs/figures/elbow_plot.png)

### Silhouette scores

![Silhouette scores](outputs/figures/silhouette_scores.png)

### Two-dimensional cluster projection

![Two-dimensional cluster projection](outputs/figures/cluster_visualization.png)

The visualization shows extensive overlap, while the first two SVD components preserve only 2.28% of total variance. The plot must therefore be interpreted together with the low silhouette scores. Both forms of evidence suggest that the complaint narratives do not form sharply separated groups under the current representation and algorithm.

## Interpreted cluster themes

| Cluster | Documents | Share | Exploratory interpretation |
| --- | --- | --- | --- |
| 0 | 361 | 18.20% | General debt, loan, mortgage, and payment disputes |
| 1 | 224 | 11.29% | Debt validation and collection-law complaints |
| 2 | 466 | 23.49% | General account, payment, and resolution problems |
| 3 | 39 | 1.97% | Repeated false credit-reporting language |
| 4 | 19 | 0.96% | Template-like formal debt-verification requests |
| 5 | 236 | 11.90% | Credit-card charges, fees, and billing disputes |
| 6 | 209 | 10.53% | Bank accounts, deposits, transfers, and fraud |
| 7 | 168 | 8.47% | Unauthorized digital-payment transactions |
| 8 | 206 | 10.38% | Credit reporting, identity theft, and inaccurate information |
| 9 | 56 | 2.82% | Wells Fargo-related account and card complaints |

These names are post-hoc interpretations, not ground-truth labels.

## Result files

All generated figures and tables are committed to the repository.

### Figures

- [`narrative_length_distribution.png`](outputs/figures/narrative_length_distribution.png)
- [`elbow_plot.png`](outputs/figures/elbow_plot.png)
- [`silhouette_scores.png`](outputs/figures/silhouette_scores.png)
- [`cluster_visualization.png`](outputs/figures/cluster_visualization.png)

### Tables

- [`dataset_summary.csv`](outputs/tables/dataset_summary.csv)
- [`k_evaluation_results.csv`](outputs/tables/k_evaluation_results.csv)
- [`cluster_sizes.csv`](outputs/tables/cluster_sizes.csv)
- [`cluster_top_terms.csv`](outputs/tables/cluster_top_terms.csv)
- [`representative_complaints.csv`](outputs/tables/representative_complaints.csv)
- [`top_products_by_cluster.csv`](outputs/tables/top_products_by_cluster.csv)
- [`top_issues_by_cluster.csv`](outputs/tables/top_issues_by_cluster.csv)

The repository also includes the full generated dataset with cluster assignments:

- [`clustered_complaints.csv`](outputs/clustered_complaints.csv)

## Repository structure

```text
consumer-complaints-kmeans-text-context/
├── .github/
│   └── workflows/
│       └── repository-validation.yml
├── data/
│   └── README.md
├── docs/
│   ├── DATA_AND_ETHICS.md
│   ├── FULL_REPORT.md
│   ├── METHODOLOGICAL_POSITION.md
│   ├── RESULTS.md
│   └── KMeans_Consumer_Complaints_Report_English.docx
├── notebooks/
│   └── consumer_complaints_kmeans.ipynb
├── outputs/
│   ├── figures/
│   ├── tables/
│   ├── clustered_complaints.csv
│   └── project_summary.json
├── scripts/
│   └── validate_repository.py
├── .gitignore
├── CITATION.cff
├── CONTRIBUTING.md
├── environment.yml
├── LICENSE
├── Makefile
├── README.md
└── requirements.txt
```

## Reproduce the analysis

### Option 1: `pip`

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab notebooks/consumer_complaints_kmeans.ipynb
```

Run all cells from top to bottom. The notebook creates the `data/` and `outputs/` directories automatically.

### Option 2: Conda

```bash
conda env create -f environment.yml
conda activate complaint-kmeans
jupyter lab notebooks/consumer_complaints_kmeans.ipynb
```

### Validate the repository package

```bash
python scripts/validate_repository.py
```

## Important limitations

- The analysis uses the first 50,000 streamed records, not a random sample of the complete database.
- Only 1,984 records contained sufficiently long, unique narratives after cleaning.
- K=10 was the upper boundary of the evaluated range.
- All silhouette scores were close to zero.
- K-means imposes one cluster per document even when a complaint contains several themes.
- Repeated legal templates can produce wording-based clusters.
- Company names can create entity-specific clusters.
- Numerical information was removed during preprocessing.
- The two-dimensional SVD view explains only a small share of total variance.
- Cluster names are interpretive and were not externally validated.

## Possible extensions

More appropriate follow-up analyses may include sentence-level segmentation, soft clustering, topic models, density-based clustering, transformer sentence embeddings, near-duplicate/template detection, stability analysis across seeds and samples, and evaluation of K beyond ten. These methods should still be selected only after clarifying the analytical objective and examining the corpus structure.

## Data and ethics

The source narratives are publicly released consumer complaints. Public availability does not remove the need for responsible use. Do not attempt to identify complainants, reconstruct masked information, or use the text for decisions about individuals. See [`docs/DATA_AND_ETHICS.md`](docs/DATA_AND_ETHICS.md).

## Documentation

- [Full GitHub-readable report](docs/FULL_REPORT.md)
- [Detailed results](docs/RESULTS.md)
- [Methodological position](docs/METHODOLOGICAL_POSITION.md)
- [Data and ethics](docs/DATA_AND_ETHICS.md)
- [Original English Word report](docs/KMeans_Consumer_Complaints_Report_English.docx)

## License

The project code and original documentation are released under the MIT License. The source dataset remains subject to the terms of its original provider.
