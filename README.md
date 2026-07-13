# When Text Context Limits Clustering

## An exploratory NLP case study using TF–IDF and K-means on consumer financial complaints

**Author:** Behzad Lajevard  
**Project type:** Unsupervised Natural Language Processing  
**Dataset:** CFPB consumer complaint narratives from Hugging Face  
**Notebook:** [`notebooks/consumer_complaints_kmeans.ipynb`](notebooks/consumer_complaints_kmeans.ipynb)  
**Executable pipeline:** [`src/pipeline.py`](src/pipeline.py)

## What this project is—and is not

This repository is **not** intended to prove that K-means works successfully on consumer complaint narratives, and it is not presented as a production-ready topic-classification system.

Its purpose is methodological: **text context, corpus structure, and problem formulation must be examined before a clustering model is selected**. Careful preprocessing can remove noise, standardize narratives, and construct valid numerical features. It cannot create naturally separated clusters when the underlying texts are multi-topic, formulaic, company-specific, or semantically overlapping.

The low silhouette scores are therefore not hidden as failures. They are the main empirical lesson. Even precise preprocessing cannot compensate for a corpus that does not support hard thematic separation. Model construction should follow a defensible research question and a diagnosis of the text—not blind algorithm selection.

## Project summary

The pipeline streams the first 50,000 records from the public CFPB complaint dataset, detects narrative and metadata columns, cleans the text, constructs TF–IDF unigram and bigram features, evaluates K-means solutions from K=2 to K=10, interprets the selected clusters, and exports analytical outputs.

| Item | Result |
|---|---:|
| Streamed source records | 50,000 |
| Usable complaint narratives | 1,984 |
| TF–IDF features | 7,374 |
| Candidate K values | 2–10 |
| Selected K within tested range | 10 |
| Best tested silhouette score | 0.0245 |
| Variance explained by two SVD components | 2.28% |

K=10 was the **upper boundary** of the tested range, and the silhouette score remained very low. It is the best configuration among the tested candidates, not proof of a globally optimal or naturally separated ten-cluster structure.

> High-quality preprocessing improves the representation of text, but it does not guarantee that the corpus is clusterable. The analytical objective, unit of analysis, and modelling strategy must be grounded in the context of the text before a model is built.

## Pipeline

1. Stream a reproducible subset from Hugging Face.
2. Detect narrative, product, issue, and company columns.
3. Remove missing, empty, very short, and duplicate narratives.
4. Apply light text preprocessing.
5. Construct normalized TF–IDF unigram and bigram features.
6. Evaluate K using inertia and sampled silhouette scores.
7. Fit the final K-means model.
8. Inspect cluster sizes and centroid terms.
9. Retrieve representative complaints nearest to each centroid.
10. Compare clusters with product and issue metadata.
11. Create a two-dimensional Truncated SVD diagnostic.
12. Export tables, figures, clustered data, and metadata locally.

## Key figures committed to the repository

### Narrative length distribution

![Narrative length distribution](outputs/figures/narrative_length_distribution.png)

### Elbow curve

![Elbow plot](outputs/figures/elbow_plot.png)

The full local pipeline also generates the silhouette-score plot and the two-dimensional cluster visualization. Their numerical results and interpretation are preserved in the committed tables and documentation.

## Interpreted cluster themes

| Cluster | Documents | Share | Exploratory interpretation |
| ---: | ---: | ---: | --- |
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

## Committed result tables

- [`dataset_summary.csv`](outputs/tables/dataset_summary.csv)
- [`k_evaluation_results.csv`](outputs/tables/k_evaluation_results.csv)
- [`cluster_sizes.csv`](outputs/tables/cluster_sizes.csv)
- [`cluster_top_terms.csv`](outputs/tables/cluster_top_terms.csv)
- [`top_products_by_cluster.csv`](outputs/tables/top_products_by_cluster.csv)
- [`top_issues_by_cluster.csv`](outputs/tables/top_issues_by_cluster.csv)
- [`project_summary.json`](outputs/project_summary.json)

Raw, cleaned, representative-document, and fully clustered narrative files are generated locally but intentionally excluded from the public repository to avoid unnecessary redistribution of complaint text.

## Repository structure

```text
consumer-complaints-kmeans-text-context/
├── data/
│   └── README.md
├── docs/
│   ├── DATA_AND_ETHICS.md
│   ├── METHODOLOGICAL_POSITION.md
│   └── RESULTS.md
├── notebooks/
│   └── consumer_complaints_kmeans.ipynb
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── project_summary.json
├── src/
│   └── pipeline.py
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

### Using `pip`

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python src/pipeline.py
```

Alternatively, open the notebook and run all cells from top to bottom.

### Using Conda

```bash
conda env create -f environment.yml
conda activate complaint-kmeans
python src/pipeline.py
```

## Important limitations

- The analysis uses the first 50,000 streamed records, not a random sample of the full database.
- Only 1,984 records contained sufficiently long, unique narratives after cleaning.
- K=10 was the upper boundary of the evaluated range.
- All silhouette scores were close to zero.
- K-means imposes one cluster per document even when a complaint contains several themes.
- Repeated legal templates can produce wording-based clusters.
- Company names can create entity-specific clusters.
- Numerical information was removed during preprocessing.
- The two-dimensional SVD view explained only a small share of total variance.
- Cluster names are interpretive and were not externally validated.

## Possible extensions

Potential follow-up analyses include sentence-level segmentation, soft clustering, topic models, density-based clustering, transformer sentence embeddings, template detection, stability analysis across seeds and samples, and evaluation of K beyond ten. These methods should still be selected only after clarifying the analytical objective and examining the corpus structure.

## Documentation

- [Detailed results](docs/RESULTS.md)
- [Methodological position](docs/METHODOLOGICAL_POSITION.md)
- [Data and ethics](docs/DATA_AND_ETHICS.md)

## License

Project code and original documentation are released under the MIT License. The source dataset remains subject to the terms of its original provider.
