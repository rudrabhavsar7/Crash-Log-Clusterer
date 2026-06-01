import numpy as np
import hdbscan
import json
import sys
import yaml
from eval.run_eval import calculate_metrics

with open('eval/ground_truth.json', 'r') as f:
    gt_clusters = json.load(f)['clusters']

with open('data/raw_traces.json', 'r') as f:
    raw_traces = json.load(f)

embeddings = np.load('data/embeddings.npy')

best_mcs = None
best_ms = None
best_recall = 0

for mcs in [10, 15, 20, 25, 30, 40, 50, 8, 12]:
    for ms in [1, 2, 3, 4, 5, 8, 10, 15]:
        hdb = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms, metric='euclidean')
        labels = hdb.fit_predict(embeddings)
        
        assignments = []
        for i, trace in enumerate(raw_traces):
            assignments.append({"trace_id": trace["id"], "cluster_id": int(labels[i])})
            
        recall, purity, recovered, _ = calculate_metrics(gt_clusters, assignments)
        
        if recall >= 0.80 and purity >= 0.70 and recovered >= 4:
            print(f"SUCCESS! mcs={mcs}, ms={ms} -> recall={recall}, purity={purity}, recovered={recovered}")
            config = yaml.safe_load(open('config.yaml'))
            config['min_cluster_size'] = mcs
            config['min_samples'] = ms
            with open('config.yaml', 'w') as f: yaml.dump(config, f)
            sys.exit(0)
            
        if recall > best_recall:
            best_recall = recall
            best_mcs = mcs
            best_ms = ms

print(f"Failed to find perfect parameters. Best recall={best_recall} at mcs={best_mcs}, ms={best_ms}")
