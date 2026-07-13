# Results

## Dataset and representation

The pipeline streamed 50,000 source records and retained 1,984 sufficiently long, unique complaint narratives after cleaning. The cleaned texts contained an average of 212.61 words and a median of 175 words. TF–IDF produced a sparse matrix of 1,984 documents by 7,374 unigram and bigram features.

![Narrative length distribution](../outputs/figures/narrative_length_distribution.png)

## Candidate cluster counts

| K | Inertia | Silhouette score |
| ---: | ---: | ---: |
| 2 | 1885.9161 | 0.011205 |
| 3 | 1856.5369 | 0.018331 |
| 4 | 1843.9412 | 0.019189 |
| 5 | 1834.8401 | 0.019856 |
| 6 | 1822.8770 | 0.021797 |
| 7 | 1821.6920 | 0.019385 |
| 8 | 1808.3790 | 0.023217 |
| 9 | 1804.9518 | 0.023306 |
| 10 | 1796.5316 | 0.024532 |

![Elbow plot](../outputs/figures/elbow_plot.png)

Inertia declined without a sharp elbow. The silhouette score reached its tested maximum at K=10, but the absolute score remained close to zero. K=10 was also the upper boundary of the search. The selected configuration is therefore exploratory and does not establish a naturally separated ten-topic structure.

## Cluster sizes and interpretation

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

Interpretation combined centroid terms with product and issue distributions. Several small clusters were driven by repeated legal wording, showing that K-means can separate documents by template similarity rather than by a distinct substantive problem.

## Main conclusion

The project produced interpretable recurring themes, but interpretability did not imply strong statistical separation. The low silhouette scores, boundary solution for K, template-driven clusters, and overlap observed in the original two-dimensional diagnostic all point to the same conclusion: the corpus does not support sharp hard-clustering boundaries under this representation.

Careful preprocessing was necessary, but it could not change the multi-topic and overlapping nature of the texts. This is the central result of the project.
