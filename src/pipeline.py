"""End-to-end exploratory clustering of CFPB consumer complaint narratives.

This pipeline is intentionally diagnostic. It demonstrates that technically
careful preprocessing cannot guarantee meaningful hard clusters when the text
corpus is multi-topic, formulaic, entity-specific, and semantically overlapping.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

CONFIG = {
    "dataset_name": "Mouwiya/cfpb-consumer-complaints",
    "download_limit": 50_000,
    "min_words": 10,
    "min_df": 5,
    "max_df": 0.90,
    "max_features": 8_000,
    "ngram_range": (1, 2),
    "k_min": 2,
    "k_max": 10,
    "silhouette_sample_size": 2_000,
    "random_state": 42,
    "top_terms_per_cluster": 15,
    "representative_documents": 3,
}

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
for directory in (DATA_DIR, OUTPUT_DIR, FIGURE_DIR, TABLE_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def detect_column(columns: list[str], candidates: list[str], keywords: list[str]) -> str | None:
    """Return the first exact or keyword-based column match."""
    for candidate in candidates:
        if candidate in columns:
            return candidate
    for column in columns:
        normalized = column.lower().replace("_", " ")
        if any(keyword in normalized for keyword in keywords):
            return column
    return None


def clean_text(text: object) -> str:
    """Apply light preprocessing while preserving useful financial vocabulary."""
    value = str(text).lower()
    value = re.sub(r"https?://\S+|www\.\S+", " ", value)
    value = re.sub(r"\S+@\S+", " ", value)
    value = re.sub(r"\b[xX]{2,}\b", " ", value)
    value = re.sub(r"[^a-zA-Z'\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def save_figure(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / name, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    print("Streaming CFPB consumer complaints from Hugging Face...")
    stream = load_dataset(CONFIG["dataset_name"], split="train", streaming=True)
    records: list[dict] = []
    for row_number, row in enumerate(stream):
        records.append(dict(row))
        if row_number + 1 >= CONFIG["download_limit"]:
            break

    if not records:
        raise RuntimeError("No records were downloaded.")

    df_raw = pd.DataFrame(records)
    df_raw.to_csv(DATA_DIR / "complaints_raw.csv", index=False)

    columns = list(df_raw.columns)
    text_column = detect_column(
        columns,
        ["consumer_complaint_narrative", "Consumer complaint narrative", "complaint_what_happened", "narrative", "text"],
        ["complaint narrative", "complaint text", "narrative"],
    )
    product_column = detect_column(columns, ["product", "Product"], ["product"])
    issue_column = detect_column(columns, ["issue", "Issue"], ["issue"])
    company_column = detect_column(columns, ["company", "Company"], ["company"])
    if text_column is None:
        raise ValueError(f"No narrative column found. Available columns: {columns}")

    df = df_raw.copy()
    df = df[df[text_column].notna()].copy()
    df[text_column] = df[text_column].astype(str).str.strip()
    df = df[df[text_column].ne("")].copy()
    df["word_count_raw"] = df[text_column].str.split().str.len()
    df = df[df["word_count_raw"] >= CONFIG["min_words"]].copy()
    df = df.drop_duplicates(subset=[text_column]).reset_index(drop=True)

    df["clean_text"] = df[text_column].map(clean_text)
    df["word_count_clean"] = df["clean_text"].str.split().str.len()
    df = df[df["word_count_clean"] >= CONFIG["min_words"]].reset_index(drop=True)
    df.to_csv(DATA_DIR / "complaints_clean.csv", index=False)

    summary = pd.DataFrame(
        {
            "metric": ["downloaded_rows", "usable_documents", "mean_words", "median_words", "minimum_words", "maximum_words"],
            "value": [
                len(df_raw),
                len(df),
                round(float(df["word_count_clean"].mean()), 2),
                round(float(df["word_count_clean"].median()), 2),
                int(df["word_count_clean"].min()),
                int(df["word_count_clean"].max()),
            ],
        }
    )
    summary.to_csv(TABLE_DIR / "dataset_summary.csv", index=False)

    plt.figure(figsize=(9, 5))
    plt.hist(df["word_count_clean"], bins=50)
    plt.xlabel("Narrative length in words")
    plt.ylabel("Frequency")
    plt.title("Distribution of Complaint Narrative Length")
    save_figure("narrative_length_distribution.png")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        min_df=CONFIG["min_df"],
        max_df=CONFIG["max_df"],
        max_features=CONFIG["max_features"],
        ngram_range=CONFIG["ngram_range"],
        sublinear_tf=True,
        norm="l2",
    )
    X_tfidf = vectorizer.fit_transform(df["clean_text"])

    evaluation_records: list[dict] = []
    sample_size = min(CONFIG["silhouette_sample_size"], len(df))
    rng = np.random.default_rng(CONFIG["random_state"])
    sample_indices = rng.choice(len(df), size=sample_size, replace=False)

    for k in range(CONFIG["k_min"], CONFIG["k_max"] + 1):
        model = KMeans(n_clusters=k, random_state=CONFIG["random_state"], n_init=20, max_iter=300)
        labels = model.fit_predict(X_tfidf)
        score = silhouette_score(X_tfidf[sample_indices], labels[sample_indices], metric="cosine")
        evaluation_records.append({"k": k, "inertia": model.inertia_, "silhouette_score": score})
        print(f"K={k}: inertia={model.inertia_:.4f}, silhouette={score:.4f}")

    evaluation_df = pd.DataFrame(evaluation_records)
    evaluation_df.to_csv(TABLE_DIR / "k_evaluation_results.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(evaluation_df["k"], evaluation_df["inertia"], marker="o")
    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia")
    plt.title("Elbow Curve for K-means")
    save_figure("elbow_plot.png")

    plt.figure(figsize=(8, 5))
    plt.plot(evaluation_df["k"], evaluation_df["silhouette_score"], marker="o")
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette score")
    plt.title("Silhouette Scores Across Candidate K Values")
    save_figure("silhouette_scores.png")

    best_row = evaluation_df.loc[evaluation_df["silhouette_score"].idxmax()]
    best_k = int(best_row["k"])
    best_silhouette = float(best_row["silhouette_score"])

    final_kmeans = KMeans(n_clusters=best_k, random_state=CONFIG["random_state"], n_init=20, max_iter=300)
    df["cluster"] = final_kmeans.fit_predict(X_tfidf)

    cluster_sizes = (
        df["cluster"].value_counts().sort_index().rename_axis("cluster").reset_index(name="document_count")
    )
    cluster_sizes["percentage"] = (cluster_sizes["document_count"] / len(df) * 100).round(2)
    cluster_sizes.to_csv(TABLE_DIR / "cluster_sizes.csv", index=False)

    feature_names = np.asarray(vectorizer.get_feature_names_out())
    top_terms_records: list[dict] = []
    for cluster_id in range(best_k):
        indices = final_kmeans.cluster_centers_[cluster_id].argsort()[::-1][: CONFIG["top_terms_per_cluster"]]
        for rank, term in enumerate(feature_names[indices], start=1):
            top_terms_records.append({"cluster": cluster_id, "rank": rank, "term": term})
    pd.DataFrame(top_terms_records).to_csv(TABLE_DIR / "cluster_top_terms.csv", index=False)

    distances = final_kmeans.transform(X_tfidf)
    representative_records: list[dict] = []
    for cluster_id in range(best_k):
        cluster_indices = np.flatnonzero(df["cluster"].to_numpy() == cluster_id)
        closest = cluster_indices[np.argsort(distances[cluster_indices, cluster_id])[: CONFIG["representative_documents"]]]
        for rank, row_index in enumerate(closest, start=1):
            record = {"cluster": cluster_id, "rank": rank, "narrative": df.loc[row_index, text_column]}
            if product_column:
                record["product"] = df.loc[row_index, product_column]
            if issue_column:
                record["issue"] = df.loc[row_index, issue_column]
            if company_column:
                record["company"] = df.loc[row_index, company_column]
            representative_records.append(record)
    pd.DataFrame(representative_records).to_csv(TABLE_DIR / "representative_complaints.csv", index=False)

    def save_top_metadata(column: str | None, filename: str) -> None:
        if column is None:
            return
        counts = df.groupby(["cluster", column]).size().reset_index(name="count")
        counts["rank"] = counts.groupby("cluster")["count"].rank(method="first", ascending=False)
        counts[counts["rank"] <= 5].sort_values(["cluster", "rank"]).to_csv(TABLE_DIR / filename, index=False)

    save_top_metadata(product_column, "top_products_by_cluster.csv")
    save_top_metadata(issue_column, "top_issues_by_cluster.csv")

    svd = TruncatedSVD(n_components=2, random_state=CONFIG["random_state"])
    X_2d = svd.fit_transform(X_tfidf)
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=df["cluster"], alpha=0.5, s=12)
    plt.xlabel("SVD component 1")
    plt.ylabel("SVD component 2")
    plt.title("Two-Dimensional Visualization of K-means Clusters")
    plt.colorbar(scatter, label="Cluster")
    save_figure("cluster_visualization.png")

    df.to_csv(OUTPUT_DIR / "clustered_complaints.csv", index=False)
    project_summary = {
        "dataset": CONFIG["dataset_name"],
        "number_of_documents": int(len(df)),
        "text_column": text_column,
        "product_column": product_column,
        "issue_column": issue_column,
        "tfidf_features": int(X_tfidf.shape[1]),
        "selected_k": best_k,
        "best_silhouette_score": round(best_silhouette, 4),
        "explained_variance_two_components": round(float(svd.explained_variance_ratio_.sum()), 4),
        "random_state": CONFIG["random_state"],
        "selection_note": "The selected K maximized the tested silhouette score, but separation remained weak and results are exploratory.",
    }
    (OUTPUT_DIR / "project_summary.json").write_text(json.dumps(project_summary, indent=4), encoding="utf-8")

    print("Pipeline completed successfully.")
    print(json.dumps(project_summary, indent=2))


if __name__ == "__main__":
    main()
