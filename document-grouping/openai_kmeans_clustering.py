"""
Document-level clustering for PCM's "supervised-clustering" variant: OpenAI
text embeddings + KMeans, works for any task registered in task.py.

Run wherever OPENAI_API_KEY is available (local machine or a Jean Zay
frontend node -- compute nodes have no internet, see slurms/jeanzay/pipeline/env.sh):

    export OPENAI_API_KEY=sk-...
    pip install openai scikit-learn numpy tqdm tiktoken   # if not already available
    python document-grouping/openai_kmeans_clustering.py --dataset legal-eval-v2

Output: document-grouping/document-groupe/supervised-clustering/<dataset>/
        {train,dev,test}.jsonl, each line {"id", "cluster", "keywords"} --
        the schema build_centroids.py / match_centroids.py expect (just pass
        --emb_type supervised-clustering to those).

Method:
  1. Embed each document's full text (ssc-datasets/<dataset>/*.json ->
     doc["data"]["text"]) with an OpenAI embedding model.
  2. KMeans (K, default 16) on the TRAIN embeddings only.
  3. Assign dev/test documents to the nearest train cluster centroid (cosine
     similarity).
  4. Per cluster, extract top keywords via TF-IDF over the member documents'
     text (highest mean TF-IDF term score in that cluster).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import tiktoken
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from openai import OpenAI

# text-embedding-3-small: 8191 tokens max per single input, 300000 tokens max
# per request (summed over all inputs in the batch). Legal judgments vary
# wildly in length (some run to tens of thousands of tokens) and tokenize
# less efficiently than average English prose (citations, numbers, names),
# so a chars/4 estimate undercounts -- use the real tokenizer instead.
MAX_TOKENS_PER_DOC = 8000            # margin below the real 8191 cap
MAX_TOKENS_PER_REQUEST = 250_000     # margin below the real 300000 cap


def doc_text(doc, encoding):
    data = doc["data"]
    text = data["text"] if isinstance(data, dict) else data
    tokens = encoding.encode(text)
    if len(tokens) > MAX_TOKENS_PER_DOC:
        text = encoding.decode(tokens[:MAX_TOKENS_PER_DOC])
    return text


def load_split(src, split, encoding):
    docs = json.load(open(src / f"{split}.json"))
    return [(str(d["id"]), doc_text(d, encoding)) for d in docs]


def make_batches(texts, encoding):
    batch, budget = [], 0
    for text in texts:
        n_tokens = max(1, len(encoding.encode(text)))
        if batch and budget + n_tokens > MAX_TOKENS_PER_REQUEST:
            yield batch
            batch, budget = [], 0
        batch.append(text)
        budget += n_tokens
    if batch:
        yield batch


def embed_texts(texts, client, model, encoding):
    batches = list(make_batches(texts, encoding))
    vecs = []
    for batch in tqdm(batches, desc="embedding", ncols=80):
        resp = client.embeddings.create(model=model, input=batch)
        vecs.extend([d.embedding for d in resp.data])
    return np.array(vecs, dtype=np.float32)


def top_keywords_per_cluster(texts, labels, k, n_keywords):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf = vectorizer.fit_transform(texts)
    terms = np.array(vectorizer.get_feature_names_out())
    keywords = {}
    for c in range(k):
        mask = labels == c
        if not mask.any():
            keywords[c] = ""
            continue
        mean_scores = np.asarray(tfidf[mask].mean(axis=0)).ravel()
        top_idx = mean_scores.argsort()[::-1][:n_keywords]
        keywords[c] = ", ".join(terms[top_idx])
    return keywords


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Task name, matching ssc-datasets/<dataset>/")
    ap.add_argument("--k", type=int, default=16, help="Number of clusters")
    ap.add_argument("--embed-model", default="text-embedding-3-small")
    ap.add_argument("--n-keywords", type=int, default=3)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--ssc-datasets-dir", default="ssc-datasets")
    ap.add_argument("--out-root", default="document-grouping/document-groupe/supervised-clustering")
    args = ap.parse_args()

    src = Path(args.ssc_datasets_dir) / args.dataset
    dst = Path(args.out_root) / args.dataset
    dst.mkdir(parents=True, exist_ok=True)

    encoding = tiktoken.encoding_for_model(args.embed_model)
    client = OpenAI()

    splits = {s: load_split(src, s, encoding) for s in ("train", "dev", "test")}

    print(f"Embedding {sum(len(v) for v in splits.values())} documents with {args.embed_model}...")
    embeddings = {s: embed_texts([t for _, t in docs], client, args.embed_model, encoding)
                 for s, docs in splits.items()}

    km = KMeans(n_clusters=args.k, random_state=args.random_state, n_init="auto")
    train_labels = km.fit_predict(embeddings["train"])
    centroids = km.cluster_centers_

    train_texts = [t for _, t in splits["train"]]
    keywords = top_keywords_per_cluster(train_texts, train_labels, args.k, args.n_keywords)

    all_labels = {"train": train_labels}
    for split in ("dev", "test"):
        sims = cosine_similarity(embeddings[split], centroids)
        all_labels[split] = sims.argmax(axis=1)

    for split, docs in splits.items():
        out_path = dst / f"{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for (doc_id, _), cluster in zip(docs, all_labels[split]):
                cluster = int(cluster)
                f.write(json.dumps({
                    "id": doc_id,
                    "cluster": cluster,
                    "keywords": keywords[cluster],
                }) + "\n")
        print(f"{split}: {len(docs)} docs -> {out_path}")

    print("\nCluster keywords:")
    for c in range(args.k):
        print(f"  {c}: {keywords[c]}")


if __name__ == "__main__":
    main()
