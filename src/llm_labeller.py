import json
import os
import time
import yaml
import numpy as np
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

def cosine_similarity(a, b):
    # a: (N, D), b: (D,)
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b)
    # prevent division by zero
    if norm_b == 0:
        return np.zeros(a.shape[0])
    norm_a[norm_a == 0] = 1.0
    return np.dot(a, b) / (norm_a * norm_b)

def call_groq(client, model, prompt_text, max_retries=1):
    system_prompt = (
        "You are a senior Android engineer. Given 3 stack traces from the same crash cluster, "
        "respond ONLY with a valid JSON object, no markdown, no explanation."
    )
    
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_text}
                ],
                temperature=0.1
            )
            content = response.choices[0].message.content.strip()
            
            # Sometimes LLMs wrap JSON in markdown blocks despite instructions
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
                
            return json.loads(content.strip())
            
        except Exception as e:
            if attempt == max_retries:
                print(f"Failed to parse JSON after {max_retries + 1} attempts: {e}")
                return {
                    "label": "Unknown",
                    "suspect_file": "Unknown",
                    "severity": "medium",
                    "category": "other"
                }

def main():
    # 1. LOAD configurations and data
    with open("../config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    groq_model = config.get("groq_model", "llama-3.1-8b-instant")
    
    with open("../data/raw_traces.json", "r") as f:
        raw_traces = json.load(f)
           
    with open("../data/clusters_hdbscan.json", "r") as f:
        hdbscan_data = json.load(f)
        
    embeddings = np.load("../data/embeddings.npy")
    
    # Initialize Groq client
    api_key = os.environ.get("GROQ_API_KEY")
    client = Groq(api_key=api_key)
    
    # Map trace_id to index and trace_id to raw_text
    trace_id_to_idx = {t["id"]: i for i, t in enumerate(raw_traces)}
    trace_id_to_text = {t["id"]: t["raw_text"] for t in raw_traces}
    
    # Group assignments by cluster
    clusters = {}
    for assignment in hdbscan_data.get("assignments", []):
        cid = assignment["cluster_id"]
        if cid == -1:
            continue
        if cid not in clusters:
            clusters[cid] = []
        clusters[cid].append(assignment["trace_id"])
        
    output_clusters = []
    total_clusters = len(clusters)
    
    print(f"Found {total_clusters} valid clusters to label.")
    
    # 2. Iterate through each cluster
    for i, (cid, trace_ids) in enumerate(sorted(clusters.items()), 1):
        print(f"Labelling cluster {i}/{total_clusters}...")
        
        # a. Find the 3 traces closest to centroid
        cluster_indices = [trace_id_to_idx[tid] for tid in trace_ids]
        cluster_embeddings = embeddings[cluster_indices]
        
        centroid = np.mean(cluster_embeddings, axis=0)
        
        # Calculate cosine similarities against all traces in the cluster
        similarities = cosine_similarity(cluster_embeddings, centroid)
        
        # Get top 3 indices (relative to cluster_embeddings)
        num_traces = min(3, len(trace_ids))
        top_indices = np.argsort(similarities)[-num_traces:][::-1]
        
        top_trace_ids = [trace_ids[idx] for idx in top_indices]
        top_texts = [trace_id_to_text[tid] for tid in top_trace_ids]
        
        # Pad texts if we have fewer than 3
        while len(top_texts) < 3:
            top_texts.append("N/A")
            
        # b. Format prompt
        prompt_text = (
            f"Analyze these 3 Android crash stack traces and identify the root cause.\n\n"
            f"Trace 1: {top_texts[0]}\n"
            f"Trace 2: {top_texts[1]}\n"
            f"Trace 3: {top_texts[2]}\n\n"
            f"Respond with exactly this JSON structure:\n"
            f"{{\n"
            f"  \"label\": \"short human-readable root cause (max 10 words)\",\n"
            f"  \"suspect_file\": \"most likely file/class responsible\",\n"
            f"  \"severity\": \"critical|high|medium|low\",\n"
            f"  \"category\": \"auth|memory|network|database|ui|io|other\"\n"
            f"}}"
        )
        
        # 3. & 4. Call LLM and parse safely
        llm_label = call_groq(client, groq_model, prompt_text)
        
        output_clusters.append({
            "cluster_id": cid,
            "trace_count": len(trace_ids),
            "representative_trace_ids": top_trace_ids,
            "llm_label": llm_label
        })
        
        # 6. Add 1-second delay
        if i < total_clusters:
            time.sleep(1)
            
    # 5. Save to labelled_clusters.json
    output_data = {
        "clusters": output_clusters
    }
    
    with open("../data/labelled_clusters.json", "w") as f:
        json.dump(output_data, f, indent=2)
        
    print("Labelling complete. Results saved to data/labelled_clusters.json")

if __name__ == "__main__":
    main()
