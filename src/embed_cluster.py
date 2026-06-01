import os
import platform

if platform.system() == "Windows":
    import ctypes
    from importlib.util import find_spec
    try:
        if (spec := find_spec("torch")) and spec.origin and os.path.exists(
            dll_path := os.path.join(os.path.dirname(spec.origin), "lib", "c10.dll")
        ):
            ctypes.CDLL(os.path.normpath(dll_path))
    except Exception:
        pass

import json
import re
import numpy as np
import yaml
import hdbscan
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import random

def preprocess(text):
    # Strip memory addresses
    text = re.sub(r'0x[0-9a-fA-F]+', '', text)
    # Strip line numbers from frames
    text = re.sub(r'\(\d+\)', '()', text)
    
    # Process line by line
    lines = text.split('\n')
    if not lines:
        return text
        
    first_line = lines[0]
    if ':' in first_line:
        exc_class, msg = first_line.split(':', 1)
        # Normalize package path: keep only last 2 segments
        parts = exc_class.split('.')
        if len(parts) > 2:
            exc_class = '.'.join(parts[-2:])
        # Lowercase exception class names
        exc_class = exc_class.lower()
        lines[0] = f"{exc_class}:{msg}"
    else:
        parts = first_line.split('.')
        if len(parts) > 2:
            first_line = '.'.join(parts[-2:])
        lines[0] = first_line.lower()

    return '\n'.join(lines)

def main():
    # Load config
    with open("../config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    random_seed = config.get('seed', 42)
    np.random.seed(random_seed)
    random.seed(random_seed)

    # 1. LOAD data/raw_traces.json
    with open("../data/raw_traces.json", "r") as f:
        raw_traces = json.load(f)

    # Reconstruct true_category mapping based on generation logic
    categories = [
        "AuthTokenExpiry", "NullPointerException", "OutOfMemoryError", 
        "NetworkTimeoutException", "DatabaseCursorException", "ANROnMainThread", 
        "ClassCastException", "IndexOutOfBoundsException", "FileNotFoundException", 
        "StackOverflowError"
    ]
    counts = [45, 55, 48, 52, 50, 42, 58, 47, 53, 50]
    trace_id_to_cat = {}
    idx = 1
    for cat, count in zip(categories, counts):
        for _ in range(count):
            trace_id_to_cat[f"trace_{idx:03d}"] = cat
            idx += 1

    # 2. PREPROCESS
    processed_texts = []
    for trace in raw_traces:
        clean_text = preprocess(trace["raw_text"])
        processed_texts.append(clean_text)

    with open("../data/processed_traces.json", "w") as f:
        json.dump(processed_texts, f, indent=2)

    # 3. EMBED
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    embeddings = model.encode(processed_texts, batch_size=32)
    np.save("../data/embeddings.npy", embeddings)

    # 4. CLUSTER HDBSCAN
    min_cluster_size = config.get('min_cluster_size', 8)
    min_samples = config.get('min_samples', 3)
    
    hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='euclidean')

    hdb_labels = hdb.fit_predict(embeddings)

    n_clusters_hdb = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)
    noise_hdb = list(hdb_labels).count(-1)

    print(f"HDBSCAN: Found {n_clusters_hdb} clusters, {noise_hdb} noise points")

    hdb_assignments = []
    for i, trace in enumerate(raw_traces):
        hdb_assignments.append({
            "trace_id": trace["id"],
            "cluster_id": int(hdb_labels[i]),
            "true_category": trace_id_to_cat[trace["id"]]
        })

    hdb_output = {
        "method": "hdbscan",
        "n_clusters": n_clusters_hdb,
        "noise_count": noise_hdb,
        "assignments": hdb_assignments
    }
    with open("../data/clusters_hdbscan.json", "w") as f:
        json.dump(hdb_output, f, indent=2)

    # 5. CLUSTER KMeans
    n_clusters_kmeans = config.get('n_clusters_kmeans', 10)
    kmeans = KMeans(n_clusters=n_clusters_kmeans, random_state=random_seed)
    kmeans_labels = kmeans.fit_predict(embeddings)

    kmeans_assignments = []
    for i, trace in enumerate(raw_traces):
        kmeans_assignments.append({
            "trace_id": trace["id"],
            "cluster_id": int(kmeans_labels[i]),
            "true_category": trace_id_to_cat[trace["id"]]
        })

    kmeans_output = {
        "method": "kmeans",
        "n_clusters": n_clusters_kmeans,
        "noise_count": 0,
        "assignments": kmeans_assignments
    }
    with open("../data/clusters_kmeans.json", "w") as f:
        json.dump(kmeans_output, f, indent=2)
        
    print("Clustering complete. Results saved to data/")

if __name__ == "__main__":
    main()
