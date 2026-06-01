import json
import os

def test_quality_gates():
    filepath = os.path.join(os.path.dirname(__file__), 'clustering.json')
    assert os.path.exists(filepath), "clustering.json must exist"
    
    with open(filepath, "r") as f:
        data = json.load(f)
        
    gates = data.get("gates_passed", {})
    
    assert gates.get("recall_gte_080") is True, f"Recall gate failed: {data['hdbscan']['recall']} < 0.80"
    assert gates.get("purity_gte_070") is True, f"Purity gate failed: {data['hdbscan']['purity']} < 0.70"
    assert gates.get("recovered_4_of_5") is True, f"Recovered gate failed: {data['hdbscan']['recovered_clusters']} < 4"
    assert gates.get("all_passed") is True, "all_passed flag should be true"
