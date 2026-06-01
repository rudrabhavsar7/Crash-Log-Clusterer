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
st.set_page_config(page_title="Crash-Log Clusterer", layout="wide", initial_sidebar_state="expanded")

# Inject Glassmorphism CSS
st.markdown("""
<style>
/* Base Theme */
.stApp {
    background: linear-gradient(135deg, #000000 0%, #121212 100%);
    background-attachment: fixed;
    color: #ffffff;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
}

/* Glassmorphism containers */
div[data-testid="stMetric"], 
div.css-1r6slb0, 
div[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.02);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 15px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.5);
}

/* Make dataframe container look glass-like */
div[data-testid="stDataFrame"] > div {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(255, 255, 255, 0.08);
}

/* Minimalist Headings */
h1, h2, h3 {
    color: #ffffff !important;
    font-weight: 300 !important;
    letter-spacing: -0.5px;
}

/* Remove default metric background to apply our glass class safely */
div[data-testid="metric-container"] {
    background: rgba(255, 255, 255, 0.02) !important;
    backdrop-filter: blur(24px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 15px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(0, 0, 0, 0.75) !important;
    backdrop-filter: blur(20px) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}
</style>
""", unsafe_allow_html=True)

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
        subprocess.run([sys.executable, "embed_cluster.py"], cwd=SRC_DIR, check=True)
    with st.spinner("Running llm_labeller.py..."):
        subprocess.run([sys.executable, "llm_labeller.py"], cwd=SRC_DIR, check=True)
    with st.spinner("Running run_eval.py..."):
        subprocess.run([sys.executable, os.path.join(ROOT_DIR, "eval", "run_eval.py")], cwd=ROOT_DIR, check=True)
    st.sidebar.success("Pipeline complete!")
    load_data.clear() # Clear Streamlit cache so new data is loaded!
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


st.title("🔍 Crash-Log Clusterer")

# ROW 1: METRICS (Bento Top Bar)
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total Traces", len(raw_traces))
m2.metric("Clusters Found", eval_data["hdbscan"]["n_clusters"])
m3.metric("Noise Points", eval_data["hdbscan"]["noise_count"])
m4.metric("Recall Gate", "✅ PASS" if eval_data["gates_passed"]["recall_gte_080"] else "❌ FAIL")
m5.metric("Purity Gate", "✅ PASS" if eval_data["gates_passed"]["purity_gte_070"] else "❌ FAIL")

st.markdown("<br>", unsafe_allow_html=True)

@st.experimental_dialog("Cluster Drill-Down", width="large")
def show_cluster_details(selected_cluster_id):
    c_info = next(c for c in labelled_clusters["clusters"] if c["cluster_id"] == selected_cluster_id)
    llm = c_info["llm_label"]
    
    st.info(f"**Label**: {llm.get('label')}  \n**File**: {llm.get('suspect_file')}  \n**Severity**: {llm.get('severity')}")
    
    c_trace_ids = []
    for assign in hdbscan_data["assignments"]:
        if assign["cluster_id"] == selected_cluster_id:
            c_trace_ids.append(assign["trace_id"])
            
    st.write(f"Showing traces for Cluster {selected_cluster_id} ({len(c_trace_ids)} traces)")
    
    for tid in c_trace_ids[:50]: # limit to 50 for performance
        t = traces_dict.get(tid)
        if not t: continue
        
        with st.expander(f"{tid} - {t.get('exception_class', '')}"):
            st.write(f"**Message**: {t.get('message', '')}")
            st.write(f"**Android Version**: {t.get('android_version', 'Unknown')}")
            frames = t.get("frames", [])[:5]
            frames_str = "\n".join([f"  at {f['file']}:{f['method']}({f['line']})" for f in frames])
            st.code(frames_str, language="java")

st.subheader("Cluster Overview")
st.markdown("<p style='color: #a0a0a0;'>Click any row to open the detailed cluster trace view.</p>", unsafe_allow_html=True)

event = st.dataframe(
    df_filtered, 
    use_container_width=True, 
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
    key="cluster_table"
)

selected_rows = event.selection.rows
if selected_rows:
    selected_idx = selected_rows[0]
    selected_cluster_id = df_filtered.iloc[selected_idx]["Cluster ID"]
    
    if st.session_state.get("last_selected_cluster") != selected_cluster_id:
        st.session_state["last_selected_cluster"] = selected_cluster_id
        show_cluster_details(selected_cluster_id)
else:
    st.session_state["last_selected_cluster"] = None

st.markdown("<br>", unsafe_allow_html=True)

# ROW 3: Charts Grid
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
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
    
    # Custom plotly layout for glassmorphism transparency
    fig_pca = px.scatter(proj_df, x="PCA1", y="PCA2", color="Cluster", hover_data=["Cluster"])
    fig_pca.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    st.plotly_chart(fig_pca, use_container_width=True)

with chart_col2:
    st.subheader("Per-Cluster Recall (HDBSCAN)")
    per_c = eval_data["hdbscan"]["per_cluster"]
    pc_df = pd.DataFrame(per_c)
    if not pc_df.empty:
        fig_recall = px.bar(pc_df, x="recall", y="gt_label", orientation="h")
        fig_recall.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig_recall, use_container_width=True)
