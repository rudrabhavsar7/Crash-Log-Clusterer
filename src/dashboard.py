# pyrefly: ignore [missing-import]
import streamlit as st
import json
import pandas as pd
import subprocess
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
import yaml
import os
import sys

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
st.set_page_config(page_title="Crash-Log Clusterer", layout="wide")

# Load data
@st.cache_data
def load_data():
    with open(os.path.join(ROOT_DIR, "data", "labelled_clusters.json")) as f:
        labelled = json.load(f)
    with open(os.path.join(ROOT_DIR, "data", "raw_traces.json")) as f:
        raw = json.load(f)
    with open(os.path.join(ROOT_DIR, "eval", "clustering.json")) as f:
        eval_data = json.load(f)
    with open(os.path.join(ROOT_DIR, "data", "clusters_hdbscan.json")) as f:
        hdbscan_clusters = json.load(f)
    embeddings = np.load(os.path.join(ROOT_DIR, "data", "embeddings.npy"))
    with open(os.path.join(ROOT_DIR, "config.yaml")) as f:
        config = yaml.safe_load(f)
    return labelled, raw, eval_data, hdbscan_clusters, embeddings, config

labelled_clusters, raw_traces, eval_data, hdbscan_data, embeddings, config = load_data()

# Sidebar
st.sidebar.header("Configuration")
min_cluster_size = st.sidebar.slider("min_cluster_size", min_value=4, max_value=50, value=config.get("min_cluster_size", 8))

if st.sidebar.button("🔄 Re-run Pipeline"):
    with st.spinner("Running embed_cluster.py..."):
        config["min_cluster_size"] = min_cluster_size
        with open(os.path.join(ROOT_DIR, "config.yaml"), "w") as f:
            yaml.dump(config, f)
        # We run the scripts with CWD = SRC_DIR because you hardcoded relative paths like `../config.yaml` inside them!
        subprocess.run([sys.executable, "embed_cluster.py"], cwd=SRC_DIR, check=True)
    with st.spinner("Running llm_labeller.py..."):
        subprocess.run([sys.executable, "llm_labeller.py"], cwd=SRC_DIR, check=True)
    st.sidebar.success("Pipeline complete!")
    st.rerun()

# Extract traces dict
traces_dict = {t["id"]: t for t in raw_traces}

# Compute scores for clusters
severity_map = {"critical": 4, "high": 3, "medium": 2, "low": 1}
cluster_rows = []
for c in labelled_clusters["clusters"]:
    llm = c["llm_label"]
    sev = llm.get("severity", "low").lower()
    score = c["trace_count"] * severity_map.get(sev, 1)
    cluster_rows.append({
        "Cluster ID": c["cluster_id"],
        "Label": llm.get("label", ""),
        "Suspect File": llm.get("suspect_file", ""),
        "Severity": sev,
        "Category": llm.get("category", ""),
        "Trace Count": c["trace_count"],
        "Score": score
    })

df_clusters = pd.DataFrame(cluster_rows)
if not df_clusters.empty:
    df_clusters = df_clusters.sort_values("Score", ascending=False).reset_index(drop=True)
    df_clusters.insert(0, "Rank", range(1, len(df_clusters) + 1))

# Sidebar Filters
all_severities = df_clusters["Severity"].unique() if not df_clusters.empty else []
all_categories = df_clusters["Category"].unique() if not df_clusters.empty else []

selected_severities = st.sidebar.multiselect("Severity", all_severities, default=all_severities)
selected_categories = st.sidebar.multiselect("Category", all_categories, default=all_categories)

if not df_clusters.empty:
    df_filtered = df_clusters[
        df_clusters["Severity"].isin(selected_severities) &
        df_clusters["Category"].isin(selected_categories)
    ]
else:
    df_filtered = pd.DataFrame()

# Header
st.title("🔍 Crash-Log Clusterer")

col1, col2, col3, col4, col5 = st.columns(5)
total_traces = len(raw_traces)
clusters_found = eval_data["hdbscan"]["n_clusters"]
noise_points = eval_data["hdbscan"]["noise_count"]
recall_pass = eval_data["gates_passed"]["recall_gte_080"]
purity_pass = eval_data["gates_passed"]["purity_gte_070"]

col1.metric("Total Traces", total_traces)
col2.metric("Clusters Found", clusters_found)
col3.metric("Noise Points", noise_points)
col4.markdown(f"**Recall Gate**: {'✅' if recall_pass else '❌'}")
col5.markdown(f"**Purity Gate**: {'✅' if purity_pass else '❌'}")

tab1, tab2, tab3 = st.tabs(["Clusters", "Comparison", "Evaluation"])

with tab1:
    st.subheader("Cluster Overview")
    
    event = st.dataframe(
        df_filtered, 
        use_container_width=True, 
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row"
    )
    
    selected_rows = event.selection.rows
    if selected_rows:
        selected_idx = selected_rows[0]
        selected_cluster_id = df_filtered.iloc[selected_idx]["Cluster ID"]
        
        st.divider()
        st.subheader("Drill-Down Panel")
        
        c_info = next(c for c in labelled_clusters["clusters"] if c["cluster_id"] == selected_cluster_id)
        llm = c_info["llm_label"]
        
        st.info(f"**Label**: {llm.get('label')} | **File**: {llm.get('suspect_file')} | **Severity**: {llm.get('severity')}")
        
        c_trace_ids = []
        for assign in hdbscan_data["assignments"]:
            if assign["cluster_id"] == selected_cluster_id:
                c_trace_ids.append(assign["trace_id"])
                
        st.write(f"Showing traces for Cluster {selected_cluster_id} ({len(c_trace_ids)} traces)")
        
        for tid in c_trace_ids:
            t = traces_dict.get(tid)
            if not t: continue
            
            with st.expander(f"{tid} - {t.get('exception_class', '')}: {t.get('message', '')}"):
                st.write(f"**Android Version**: {t.get('android_version', 'Unknown')}")
                frames = t.get("frames", [])[:5]
                frames_str = "\n".join([f"  at {f['file']}:{f['method']}({f['line']})" for f in frames])
                st.code(frames_str, language="java")

with tab2:
    st.subheader("HDBSCAN vs KMeans")
    
    h_data = eval_data["hdbscan"]
    k_data = eval_data["kmeans"]
    
    comp_df = pd.DataFrame({
        "Metric": ["Clusters", "Noise", "Recall", "Purity"],
        "HDBSCAN": [h_data["n_clusters"], h_data["noise_count"], h_data["recall"], h_data["purity"]],
        "KMeans": [k_data["n_clusters"], 0, k_data["recall"], k_data["purity"]]
    })
    st.table(comp_df)
    
    if not df_clusters.empty:
        fig_bar = px.bar(df_clusters, x="Cluster ID", y="Trace Count", title="HDBSCAN Cluster Sizes")
        fig_bar.update_xaxes(type='category')
        st.plotly_chart(fig_bar)
        
    st.subheader("Embedding Projection (PCA)")
    pca = PCA(n_components=2)
    proj = pca.fit_transform(embeddings)
    
    proj_df = pd.DataFrame(proj, columns=["PCA1", "PCA2"])
    
    labels = []
    for t in raw_traces:
        cid = -1
        for a in hdbscan_data["assignments"]:
            if a["trace_id"] == t["id"]:
                cid = a["cluster_id"]
                break
        labels.append(str(cid))
        
    proj_df["Cluster"] = labels
    
    fig_pca = px.scatter(proj_df, x="PCA1", y="PCA2", color="Cluster", hover_data=["Cluster"], title="2D PCA of Traces")
    st.plotly_chart(fig_pca)

with tab3:
    st.subheader("Clustering Evaluation")
    st.json(eval_data)
    
    st.subheader("Per-Cluster Recall (HDBSCAN)")
    per_c = eval_data["hdbscan"]["per_cluster"]
    pc_df = pd.DataFrame(per_c)
    if not pc_df.empty:
        fig_recall = px.bar(pc_df, x="recall", y="gt_label", orientation="h", title="Recall by Ground Truth Label")
        st.plotly_chart(fig_recall)
