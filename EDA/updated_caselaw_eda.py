"""
Exploratory Data Analysis (EDA) for the Caselaw Access Project Embeddings Dataset
Dataset: https://huggingface.co/datasets/free-law/Caselaw_Access_Project_embeddings

Uses the HuggingFace Datasets Rows API, fetches only the rows we need
over HTTPS in small JSON batches.

Run:
    pip install pandas matplotlib seaborn scikit-learn wordcloud requests
    python updated_caselaw_eda.py
"""

import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import time
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

# constants
SAMPLE_SIZE   = 2_500
BATCH_SIZE    = 100      # rows per API request (max allowed is 100)
RANDOM_STATE  = 42
N_CLUSTERS    = 8
TSNE_SAMPLE   = 2_000
OUTPUT_PREFIX = "caselaw_eda"

np.random.seed(RANDOM_STATE)

# loading data via HuggingFace Datasets Rows API
print("=" * 60)
print("1. Fetching dataset sample via HF Rows API …")
print("=" * 60)

DATASET      = "free-law/Caselaw_Access_Project_embeddings"
SPLIT        = "train"
TOTAL_ROWS   = 2_000_000   # approximate total rows in dataset
API_URL      = f"https://datasets-server.huggingface.co/rows?dataset={DATASET}&config=default&split={SPLIT}"

# Generate random offsets spread across the full dataset
# Each batch fetches BATCH_SIZE rows starting at a random position
n_batches      = (SAMPLE_SIZE // BATCH_SIZE) + 1
random_offsets = sorted(np.random.randint(0, TOTAL_ROWS - BATCH_SIZE, size=n_batches))
print(f"  Sampling {n_batches} random offsets spread across {TOTAL_ROWS:,} total rows …")

records = []

for offset in random_offsets:
    if len(records) >= SAMPLE_SIZE:
        break

    url      = f"{API_URL}&offset={int(offset)}&length={BATCH_SIZE}"
    response = requests.get(url, timeout=60)

    if response.status_code == 429:
        if records:
            print(f"\n  Rate limited after {len(records):,} rows — using what we have.")
            break
        print(f"\n  Rate limited — waiting 30 seconds …")
        time.sleep(30)
        continue

    if response.status_code != 200:
        print(f"\n  API error {response.status_code} at offset {offset}, skipping.")
        continue

    data = response.json()
    rows = data.get("rows", [])
    for r in rows:
        row = r["row"]
        records.append({"text": row["text"], "embeddings": row["embeddings"]})

    print(f"  Fetched {len(records):,} / {SAMPLE_SIZE:,} rows (offset {offset:,}) …", end="\r")
    time.sleep(0.5)

actual_size = len(records)
df = pd.DataFrame(records[:actual_size])
print(f"\n  Done. Loaded {len(df):,} records (target was {SAMPLE_SIZE:,}).\n")

# Update globals to reflect actual sample
TSNE_SAMPLE = min(TSNE_SAMPLE, len(df))

#basic stats
print("=" * 60)
print("2. Basic Statistics")
print("=" * 60)

df["char_len"]   = df["text"].str.len()
df["word_count"] = df["text"].str.split().str.len()
df["sent_count"] = df["text"].str.count(r'[.!?]+')

embed_matrix = np.vstack(df["embeddings"].values)

print(f"\n  Text length (chars):")
print(df["char_len"].describe().to_string())
print(f"\n  Word count:")
print(df["word_count"].describe().to_string())
print(f"\n  Embedding matrix shape : {embed_matrix.shape}")
print(f"  Embedding value range  : [{embed_matrix.min():.4f}, {embed_matrix.max():.4f}]")
print(f"  Mean L2 norm           : {np.linalg.norm(embed_matrix, axis=1).mean():.4f}\n")

#visualizations

# text-length distributions
print("Plotting text-length distributions …")
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, col, label, color in zip(
    axes,
    ["char_len", "word_count", "sent_count"],
    ["Character count", "Word count", "Sentence count"],
    ["steelblue", "darkorange", "seagreen"],
):
    data = df[col].clip(upper=df[col].quantile(0.99))
    ax.hist(data, bins=60, color=color, edgecolor="white", linewidth=0.4)
    ax.set_title(label, fontweight="bold")
    ax.set_xlabel(label)
    ax.set_ylabel("Frequency")
    med = df[col].median()
    ax.axvline(med, color="black", linestyle="--", linewidth=1.2,
               label=f"Median: {med:,.0f}")
    ax.legend(fontsize=9)
fig.suptitle("Text Length Distributions — Caselaw Access Project",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_PREFIX}_01_text_length.png", dpi=150)
plt.close()
print(f"  → Saved {OUTPUT_PREFIX}_01_text_length.png")

#word cloud
print("Building word cloud …")
LEGAL_STOPWORDS = {
    "the","of","and","to","in","a","that","is","for","on","was","were",
    "with","this","at","by","from","an","be","as","it","are","have",
    "has","had","not","he","she","they","we","or","but","its","their",
    "which","who","been","would","could","also","such","may","any",
    "said","court","case","plaintiff","defendant","v","no","one","upon",
    "shall","will","all","both","than","if","when","however","under",
    "mr","ms","inc","co","llc","corp","pp","st","dr",
}
all_text  = " ".join(df["text"].sample(min(500, len(df))).values)
all_text  = re.sub(r'[^a-zA-Z\s]', ' ', all_text).lower()
tokens    = [w for w in all_text.split() if w not in LEGAL_STOPWORDS and len(w) > 3]
word_freq = Counter(tokens)

wc = WordCloud(
    width=1200, height=500, background_color="white",
    colormap="Blues_r", max_words=120, prefer_horizontal=0.8,
).generate_from_frequencies(word_freq)
fig, ax = plt.subplots(figsize=(14, 5))
ax.imshow(wc, interpolation="bilinear")
ax.axis("off")
ax.set_title("Most Frequent Terms in Caselaw Texts",
             fontsize=14, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(f"{OUTPUT_PREFIX}_02_wordcloud.png", dpi=150)
plt.close()
print(f"  → Saved {OUTPUT_PREFIX}_02_wordcloud.png")

# top-30 most common terms
print("Plotting top-30 terms …")
top30 = word_freq.most_common(30)
words, counts = zip(*top30)
fig, ax = plt.subplots(figsize=(14, 5))
ax.barh(words[::-1], counts[::-1], color=sns.color_palette("Blues_r", 30))
ax.set_xlabel("Frequency")
ax.set_title("Top 30 Most Frequent Legal Terms", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_PREFIX}_03_top30_terms.png", dpi=150)
plt.close()
print(f"  → Saved {OUTPUT_PREFIX}_03_top30_terms.png")

#embedding dimension statistics
print("Plotting embedding dimension statistics …")
dim_means = embed_matrix.mean(axis=0)
dim_stds  = embed_matrix.std(axis=0)
dim_idx   = np.arange(768)
fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
axes[0].plot(dim_idx, dim_means, linewidth=0.7, color="steelblue")
axes[0].fill_between(dim_idx, dim_means - dim_stds, dim_means + dim_stds,
                     alpha=0.25, color="steelblue")
axes[0].set_ylabel("Mean ± Std")
axes[0].set_title("Embedding Dimension Means (with ±1 SD band)", fontweight="bold")
axes[1].plot(dim_idx, dim_stds, linewidth=0.7, color="darkorange")
axes[1].set_ylabel("Standard Deviation")
axes[1].set_xlabel("Embedding Dimension")
axes[1].set_title("Embedding Dimension Standard Deviations", fontweight="bold")
fig.suptitle("Per-Dimension Statistics across 768-D Embeddings",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_PREFIX}_04_embedding_dims.png", dpi=150)
plt.close()
print(f"  → Saved {OUTPUT_PREFIX}_04_embedding_dims.png")

#l2-norm distribution
print("Plotting L2-norm distribution …")
norms = np.linalg.norm(embed_matrix, axis=1)
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(norms, bins=60, color="mediumseagreen", edgecolor="white", linewidth=0.4)
ax.axvline(norms.mean(), color="black", linestyle="--", linewidth=1.4,
           label=f"Mean: {norms.mean():.2f}")
ax.set_xlabel("L2 Norm"); ax.set_ylabel("Frequency")
ax.set_title("Distribution of Embedding L2 Norms", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_PREFIX}_05_l2_norms.png", dpi=150)
plt.close()
print(f"  → Saved {OUTPUT_PREFIX}_05_l2_norms.png")

#pca projection + explained variance
print("Running PCA …")
pca    = PCA(n_components=50, random_state=RANDOM_STATE)
pca_50 = pca.fit_transform(embed_matrix)
pca_2d = pca_50[:, :2]

color_val = np.log1p(df["word_count"].values)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sc = axes[0].scatter(pca_2d[:, 0], pca_2d[:, 1],
                     c=color_val, cmap="viridis", s=5, alpha=0.6)
plt.colorbar(sc, ax=axes[0], label="log(word count)")
axes[0].set_title("PCA 2-D Projection (coloured by log word count)", fontweight="bold")
axes[0].set_xlabel("PC 1"); axes[0].set_ylabel("PC 2")
cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
axes[1].plot(range(1, 51), cumvar, marker="o", markersize=3,
             linewidth=1.5, color="steelblue")
axes[1].axhline(80, color="red",    linestyle="--", linewidth=1, label="80% threshold")
axes[1].axhline(90, color="orange", linestyle="--", linewidth=1, label="90% threshold")
axes[1].set_xlabel("Number of Principal Components")
axes[1].set_ylabel("Cumulative Explained Variance (%)")
axes[1].set_title("PCA Explained Variance", fontweight="bold")
axes[1].legend()
fig.suptitle("PCA of 768-D Caselaw Embeddings", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_PREFIX}_06_pca.png", dpi=150)
plt.close()
print(f"  → Saved {OUTPUT_PREFIX}_06_pca.png")

#kmeans clustering
print(f"Running KMeans (k={N_CLUSTERS}) …")
kmeans         = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
cluster_labels = kmeans.fit_predict(pca_50)
df["cluster"]  = cluster_labels

palette = sns.color_palette("tab10", N_CLUSTERS)
fig, ax = plt.subplots(figsize=(10, 7))
for k in range(N_CLUSTERS):
    mask = cluster_labels == k
    ax.scatter(pca_2d[mask, 0], pca_2d[mask, 1],
               s=6, alpha=0.55, color=palette[k], label=f"Cluster {k}")
ax.set_title(f"KMeans Clusters (k={N_CLUSTERS}) in PCA-2D Space", fontweight="bold")
ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
ax.legend(markerscale=2, fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUTPUT_PREFIX}_07_kmeans_clusters.png", dpi=150)
plt.close()
print(f"  → Saved {OUTPUT_PREFIX}_07_kmeans_clusters.png")

cluster_sizes = df["cluster"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(cluster_sizes.index.astype(str), cluster_sizes.values,
       color=palette, edgecolor="white")
ax.set_xlabel("Cluster ID"); ax.set_ylabel("Number of Cases")
ax.set_title("Cases per KMeans Cluster", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_PREFIX}_08_cluster_sizes.png", dpi=150)
plt.close()
print(f"  → Saved {OUTPUT_PREFIX}_08_cluster_sizes.png")

#tsne projection coloured by cluster
print(f"Running t-SNE on {TSNE_SAMPLE} samples (takes ~1–2 min) …")
tsne_idx    = np.random.choice(len(pca_50), size=min(TSNE_SAMPLE, len(pca_50)), replace=False)
tsne_input  = pca_50[tsne_idx]
tsne_labels = cluster_labels[tsne_idx]
tsne    = TSNE(n_components=2, perplexity=40, max_iter=1000,
               random_state=RANDOM_STATE, n_jobs=-1)
tsne_2d = tsne.fit_transform(tsne_input)
fig, ax = plt.subplots(figsize=(10, 7))
for k in range(N_CLUSTERS):
    mask = tsne_labels == k
    ax.scatter(tsne_2d[mask, 0], tsne_2d[mask, 1],
               s=6, alpha=0.55, color=palette[k], label=f"Cluster {k}")
ax.set_title("t-SNE of Caselaw Embeddings (coloured by KMeans cluster)", fontweight="bold")
ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
ax.legend(markerscale=2, fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUTPUT_PREFIX}_09_tsne.png", dpi=150)
plt.close()
print(f"  → Saved {OUTPUT_PREFIX}_09_tsne.png")

#cosine similarity heatmap + distribution
print("Computing cosine-similarity heatmap …")
HEAT_N   = 300
heat_idx = np.random.choice(len(embed_matrix), size=HEAT_N, replace=False)
heat_emb = embed_matrix[heat_idx]
cos_sim  = cosine_similarity(heat_emb)
fig, ax  = plt.subplots(figsize=(8, 7))
im = ax.imshow(cos_sim, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax, label="Cosine Similarity")
ax.set_title(f"Pairwise Cosine Similarity ({HEAT_N} random cases)", fontweight="bold")
ax.set_xlabel("Case index"); ax.set_ylabel("Case index")
plt.tight_layout()
plt.savefig(f"{OUTPUT_PREFIX}_10_cosine_heatmap.png", dpi=150)
plt.close()
print(f"  → Saved {OUTPUT_PREFIX}_10_cosine_heatmap.png")

triu_vals = cos_sim[np.triu_indices(HEAT_N, k=1)]
fig, ax   = plt.subplots(figsize=(8, 4))
ax.hist(triu_vals, bins=80, color="slateblue", edgecolor="white", linewidth=0.3)
ax.axvline(triu_vals.mean(), color="black", linestyle="--",
           label=f"Mean: {triu_vals.mean():.3f}")
ax.set_xlabel("Cosine Similarity"); ax.set_ylabel("Pair Count")
ax.set_title("Distribution of Pairwise Cosine Similarities", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_PREFIX}_11_cosine_distribution.png", dpi=150)
plt.close()
print(f"  → Saved {OUTPUT_PREFIX}_11_cosine_distribution.png")

#word count by cluster
print("Plotting word count by cluster …")
fig, ax = plt.subplots(figsize=(10, 5))
df.boxplot(column="word_count", by="cluster", ax=ax,
           flierprops=dict(marker=".", markersize=2, alpha=0.3),
           patch_artist=True)
plt.suptitle("")
ax.set_title("Word Count Distribution by Cluster", fontweight="bold")
ax.set_xlabel("Cluster ID"); ax.set_ylabel("Word Count")
ax.set_ylim(0, df["word_count"].quantile(0.97))
plt.tight_layout()
plt.savefig(f"{OUTPUT_PREFIX}_12_wordcount_by_cluster.png", dpi=150)
plt.close()
print(f"  → Saved {OUTPUT_PREFIX}_12_wordcount_by_cluster.png")

#summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
summary = {
    "Total records in dataset"       : "~2,000,000",
    "Sample size used for EDA"       : f"{len(df):,}",
    "Embedding dimensionality"       : 768,
    "Mean word count"                : f"{df['word_count'].mean():.0f}",
    "Median word count"              : f"{df['word_count'].median():.0f}",
    "Max word count"                 : f"{df['word_count'].max():,}",
    "Mean L2 norm of embeddings"     : f"{norms.mean():.4f}",
    "Avg pairwise cosine similarity" : f"{triu_vals.mean():.4f}",
    "PCA dims for 80% variance"      : f"{np.searchsorted(cumvar, 80) + 1}",
    "PCA dims for 90% variance"      : f"{np.searchsorted(cumvar, 90) + 1}",
    "KMeans clusters"                : N_CLUSTERS,
}
for k, v in summary.items():
    print(f"  {k:<40} {v}")

print("\nAll plots saved. EDA complete.")