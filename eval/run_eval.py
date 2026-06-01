import json
import datetime
import sys

# Windows stdout encoding fix for box drawing characters
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def calculate_metrics(gt_clusters, pred_assignments):
    # Group predicted traces by cluster_id
    pred_clusters = {}
    for assignment in pred_assignments:
        cid = assignment["cluster_id"]
        # Skip noise for evaluation
        if cid == -1:
            continue
        if cid not in pred_clusters:
            pred_clusters[cid] = set()
        pred_clusters[cid].add(assignment["trace_id"])
        
    per_cluster_results = []
    
    # Calculate Recall
    recall_sum = 0.0
    recovered_count = 0
    
    for gt in gt_clusters:
        gt_label = gt["label"]
        gt_set = set(gt["trace_ids"])
        
        max_overlap = 0
        best_p_cid = None
        
        for p_cid, p_set in pred_clusters.items():
            overlap = len(gt_set.intersection(p_set))
            if overlap > max_overlap:
                max_overlap = overlap
                best_p_cid = p_cid
                
        recall = max_overlap / len(gt_set) if len(gt_set) > 0 else 0.0
        recall_sum += recall
        
        if recall >= 0.70:
            recovered_count += 1
            
        per_cluster_results.append({
            "gt_label": gt_label,
            "recall": round(recall, 2),
            "matched_cluster_id": best_p_cid
        })
        
    avg_recall = recall_sum / len(gt_clusters) if gt_clusters else 0.0
    
    # Calculate Purity
    purity_sum = 0.0
    relevant_p_count = 0
    
    for p_cid, p_set in pred_clusters.items():
        # check if it overlaps with ANY gt cluster
        max_overlap = 0
        for gt in gt_clusters:
            gt_set = set(gt["trace_ids"])
            overlap = len(p_set.intersection(gt_set))
            if overlap > max_overlap:
                max_overlap = overlap
                
        if max_overlap > 0:
            purity = max_overlap / len(p_set)
            purity_sum += purity
            relevant_p_count += 1
            
    avg_purity = purity_sum / relevant_p_count if relevant_p_count > 0 else 0.0
    
    return round(avg_recall, 2), round(avg_purity, 2), recovered_count, per_cluster_results

def main():
    with open("eval/ground_truth.json", "r") as f:
        gt_data = json.load(f)
    gt_clusters = gt_data["clusters"]
    
    with open("data/clusters_hdbscan.json", "r") as f:
        hdbscan_data = json.load(f)
        
    with open("data/clusters_kmeans.json", "r") as f:
        kmeans_data = json.load(f)
        
    # HDBSCAN metrics
    h_recall, h_purity, h_recovered, h_per_cluster = calculate_metrics(gt_clusters, hdbscan_data["assignments"])
    
    # KMeans metrics
    k_recall, k_purity, k_recovered, k_per_cluster = calculate_metrics(gt_clusters, kmeans_data["assignments"])
    
    # Evaluate gates
    recall_gte_080 = h_recall >= 0.80
    purity_gte_070 = h_purity >= 0.70
    recovered_4_of_5 = h_recovered >= 4
    all_passed = recall_gte_080 and purity_gte_070 and recovered_4_of_5
    
    output = {
        "seed": 42,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hdbscan": {
            "n_clusters": hdbscan_data["n_clusters"],
            "noise_count": hdbscan_data["noise_count"],
            "recall": h_recall,
            "purity": h_purity,
            "recovered_clusters": h_recovered,
            "per_cluster": h_per_cluster
        },
        "kmeans": {
            "n_clusters": kmeans_data["n_clusters"],
            "recall": k_recall,
            "purity": k_purity
        },
        "gates_passed": {
            "recall_gte_080": recall_gte_080,
            "purity_gte_070": purity_gte_070,
            "recovered_4_of_5": recovered_4_of_5,
            "all_passed": all_passed
        }
    }
    
    with open("eval/clustering.json", "w") as f:
        json.dump(output, f, indent=2)
        
    # Print summary table
    print("┌─────────────────────────────────────────┐")
    print("│ QUALITY GATES EVALUATION                │")
    print("├──────────────────┬──────────┬───────────┤")
    print("│ Metric           │ Value    │ Status    │")
    print("├──────────────────┼──────────┼───────────┤")
    
    recall_status = "✅ PASS" if recall_gte_080 else "❌ FAIL"
    purity_status = "✅ PASS" if purity_gte_070 else "❌ FAIL"
    recov_status = "✅ PASS" if recovered_4_of_5 else "❌ FAIL"
    
    print(f"│ Cluster Recall   │ {h_recall:<8} │ {recall_status:<9} │")
    print(f"│ Purity           │ {h_purity:<8} │ {purity_status:<9} │")
    print(f"│ Recovered (4/5)  │ {h_recovered}/5      │ {recov_status:<9} │")
    print("└──────────────────┴──────────┴───────────┘")

if __name__ == "__main__":
    main()
